"""Command line interface."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from . import __version__

app = typer.Typer(
    help="Compute and inspect dust attenuation tables for simple galactic geometries.",
    no_args_is_help=True,
)

ConfigArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="A campaign configuration, in YAML.")
]


@app.callback()
def main() -> None:
    """Compute and inspect dust attenuation tables for simple galactic geometries."""


@app.command()
def version() -> None:
    """Report the installed version."""
    typer.echo(__version__)


@app.command()
def validate(config: ConfigArgument) -> None:
    """Check a campaign configuration and describe what it would run.

    Reads the configuration, expands every range into an axis, and reports the
    models that would result. Needs no Hyperion, so it is the quickest way to
    see whether a campaign is the size you expected before committing a cluster
    to it.
    """
    from .campaign import Campaign
    from .config import load_campaign

    campaign = Campaign(load_campaign(str(config)))
    settings = campaign.config
    typer.echo(f"{settings.label}")
    if settings.description:
        typer.echo(f"  {settings.description}")
    grains = settings.dust.file or f"Ferrara/Gordon {settings.dust.ferrara}"
    typer.echo(f"  grains       : {grains}")
    typer.echo(
        f"  grid         : {settings.geometry.radial_cells}"
        f" x {settings.geometry.vertical_cells} cells,"
        f" {settings.geometry.spacing} spacing,"
        f" {settings.geometry.sampling} sampling,"
        f" cut off {settings.geometry.cut_off}"
    )
    typer.echo(
        f"  tabulated at : {campaign.wavelengths.size} wavelengths"
        f" from {campaign.wavelengths.min():g} to {campaign.wavelengths.max():g} micron,"
        f" {campaign.inclinations.size} inclinations"
    )
    typer.echo(
        f"  Monte Carlo  : {settings.tabulation.photons} photons per wavelength and stage,"
        f" {settings.tabulation.seed_policy} seed policy"
    )
    typer.echo("  axes         :")
    for axis in campaign.axes:
        typer.echo(
            f"    {axis.name:<36} {len(axis):>5} values  from {axis.values.min():g} to {axis.values.max():g}"
        )
    typer.echo("  emitters     :")
    for emitter in campaign.emitters:
        shape = campaign.shape_for(emitter)
        names = ", ".join(axis.name for axis in campaign.axes_for(emitter)) or "none"
        count = 1
        for length in shape:
            count *= length
        typer.echo(f"    {emitter:<36} {count:>5} models  over {names}")
    typer.echo(f"  total        : {len(campaign)} models")


@app.command()
def build(
    config: ConfigArgument,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory to write model input files into.")
    ] = Path("models"),
    limit: Annotated[int, typer.Option(help="Write at most this many models; 0 writes all of them.")] = 0,
    overwrite: Annotated[bool, typer.Option(help="Rewrite inputs which already exist.")] = False,
) -> None:
    """Write the Hyperion input files for a campaign.

    Needs Hyperion, and the dust file or grain type named in the configuration.
    Existing inputs are left alone unless ``--overwrite`` is given, so an
    interrupted build can be resumed by running it again.
    """
    from .campaign import Campaign
    from .config import load_campaign
    from .dust import ferrara, load_dust
    from .hyperion_model import write_model

    campaign = Campaign(load_campaign(str(config)))
    settings = campaign.config.dust
    if settings.file is not None:
        grains = load_dust(settings.file)
    else:
        assert settings.ferrara is not None  # guaranteed by the configuration
        grains = ferrara.build(settings.ferrara, settings.reference_opacity)

    output.mkdir(parents=True, exist_ok=True)
    options = dict(campaign.grid_options())
    written = skipped = 0
    for run in campaign.runs():
        if limit and written >= limit:
            break
        path = output / f"{run.file_stem}.hdf5"
        if path.exists() and not overwrite:
            skipped += 1
            continue
        write_model(
            run.spec,
            grains,
            str(path),
            sampling=campaign.config.geometry.sampling,
            **options,
        )
        written += 1
    typer.echo(f"wrote {written} model inputs to {output}" + (f", skipped {skipped}" if skipped else ""))


@app.command()
def run(
    config: ConfigArgument,
    models: Annotated[
        Path, typer.Option("--models", "-m", help="Directory holding the model input files.")
    ] = Path("models"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory to write solved models into.")
    ] = Path("output"),
    backend: Annotated[
        str, typer.Option("--scheduler", "-s", help="Where to run: local or slurm.")
    ] = "local",
    concurrency: Annotated[int, typer.Option(help="How many models to have in flight at once.")] = 1,
    tasks: Annotated[int, typer.Option(help="MPI tasks per model; one uses the serial solver.")] = 1,
    nodes: Annotated[int, typer.Option(help="Nodes per model (slurm).")] = 1,
    partition: Annotated[str, typer.Option(help="Slurm partition.")] = "",
    walltime: Annotated[str, typer.Option(help="Slurm time limit, as HH:MM:SS.")] = "",
    memory_per_cpu: Annotated[
        int, typer.Option(help="Megabytes per cpu (slurm); 0 leaves it to the site default.")
    ] = 0,
    limit: Annotated[int, typer.Option(help="Run at most this many models; 0 runs all.")] = 0,
) -> None:
    """Solve a campaign's models.

    Needs the model inputs to exist already, from ``build``, and a Hyperion
    solver binary on ``PATH``. It needs neither the Hyperion Python package nor
    a dust file, since everything physical is baked into the inputs.

    Models whose output already exists are skipped, so an interrupted campaign
    is resumed by running this again.
    """
    from .campaign import Campaign
    from .config import load_campaign
    from .runner import (
        SCHEDULERS,
        Job,
        Resources,
        is_solved,
        scheduler,
        solver_command,
        wait_for_solved,
    )

    if backend not in SCHEDULERS:
        typer.echo(
            f"unknown scheduler {backend!r}; known schedulers are {', '.join(sorted(SCHEDULERS))}",
            err=True,
        )
        raise typer.Exit(code=1)

    campaign = Campaign(load_campaign(str(config)))
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"

    if tasks % nodes:
        typer.echo(
            f"--tasks {tasks} does not divide evenly among --nodes {nodes}. Give a task "
            f"count which is a multiple of the node count, such as {nodes * (tasks // nodes + 1)}.",
            err=True,
        )
        raise typer.Exit(code=1)
    resources = Resources(
        nodes=nodes,
        tasks_per_node=tasks // nodes,
        partition=partition or None,
        walltime=walltime or None,
        memory_per_cpu=memory_per_cpu or None,
    )

    jobs, missing, done = [], [], 0
    for candidate in campaign.runs():
        if limit and len(jobs) >= limit:
            break
        source = models / f"{candidate.file_stem}.hdf5"
        result = output / f"{candidate.file_stem}.rtout"
        if is_solved(result):
            done += 1
            continue
        if not source.exists():
            missing.append(source)
            continue
        jobs.append(
            Job(
                label=candidate.file_stem,
                command=solver_command(source, result, tasks=resources.tasks),
                log_file=logs / f"{candidate.file_stem}.log",
                resources=resources,
            )
        )

    if missing:
        typer.echo(
            f"{len(missing)} model inputs are missing, the first being {missing[0]}. "
            "Run `dust-compendium build` first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not jobs:
        typer.echo(f"nothing to do: all {done} models are already solved")
        return

    typer.echo(f"running {len(jobs)} models on {backend}" + (f", {done} already solved" if done else ""))
    finished = 0

    def report(result) -> None:
        nonlocal finished
        finished += 1
        mark = "ok " if result.succeeded else "FAILED"
        typer.echo(f"  [{finished}/{len(jobs)}] {mark} {result.job.label}")

    results = scheduler(backend, concurrency=concurrency).run(jobs, on_complete=report)
    # A zero exit status is not proof of a solve: Hyperion aborts on some
    # conditions and still exits zero, leaving an output with no SEDs in it.
    # Conversely, a shared filesystem can briefly hide groups which a completed
    # compute job has just written, so allow one bounded visibility window for
    # every successful result together before diagnosing an abort.
    successful_paths = {output / f"{result.job.label}.rtout" for result in results if result.succeeded}
    immediately_visible = {path for path in successful_paths if is_solved(path)}
    delayed = successful_paths - immediately_visible
    if delayed:
        typer.echo(
            f"waiting for {len(delayed)} completed output file"
            f"{'s' if len(delayed) != 1 else ''} to become visible"
        )
    confirmed = immediately_visible | wait_for_solved(tuple(delayed))
    aborted = [
        result
        for result in results
        if result.succeeded and output / f"{result.job.label}.rtout" not in confirmed
    ]
    for result in aborted:
        typer.echo(
            f"  {result.job.label}: the solver exited cleanly but wrote no SEDs; see {result.job.log_file}",
            err=True,
        )
    failures = [result for result in results if not result.succeeded] + aborted
    typer.echo(f"{len(results) - len(failures)} of {len(results)} models succeeded")
    for failure in failures:
        typer.echo(f"  {failure.failure_message()}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def collect(
    config: ConfigArgument,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory holding the solved models.")
    ] = Path("output"),
    tabulation: Annotated[
        Path, typer.Option("--tabulation", "-t", help="File to write the tabulation to.")
    ] = Path("attenuations.hdf5"),
) -> None:
    """Assemble solved models into a tabulation.

    Divides each model by the one with no dust, fits the high optical depth
    extrapolation, and writes the result. Can be run again at any time: it reads
    the solved models from disk and holds no state of its own.
    """
    from .campaign import Campaign
    from .config import load_campaign
    from .dust import ferrara, load_dust, opacity_to_extinction
    from .postprocess import collect as assemble

    campaign = Campaign(load_campaign(str(config)))
    grains = campaign.config.dust
    if grains.file is not None:
        dust = load_dust(grains.file)
    else:
        assert grains.ferrara is not None  # guaranteed by the configuration
        dust = ferrara.build(grains.ferrara, grains.reference_opacity)

    metadata = {
        "description": campaign.config.description,
        # Keep all fixed as well as varying assumptions. A future row-oriented
        # emulator dataset can therefore repeat the constant geometry columns
        # without having to reconstruct them from a file name or publication.
        "configuration": campaign.config.model_dump_json(by_alias=True),
    }
    if grains.description:
        metadata["dustDescription"] = grains.description
    for name, component in campaign.config.geometry.components.items():
        for role, profile in (("stellar", component.stellar), ("dust", component.dust)):
            if profile is not None:
                # Recorded so a file says which profiles it was computed for.
                # Galacticus keeps a hard-coded table of published file names
                # precisely because the files never said.
                metadata[f"{name}{role.capitalize()}Profile"] = profile.profile
    metadata["spacing"] = campaign.config.geometry.spacing
    metadata["sampling"] = campaign.config.geometry.sampling
    metadata["cutOff"] = campaign.config.geometry.cut_off

    result = assemble(campaign, output, opacity_to_extinction(dust), metadata=metadata)
    result.write(str(tabulation))
    typer.echo(f"wrote format version {result.format_version} tabulation to {tabulation}")
    for emitter in result.emitters:
        worst = result.residual[emitter]
        finite = worst[np.isfinite(worst)] if worst.size else worst
        scatter = f"{finite.max():.3g}" if finite.size else "n/a"
        typer.echo(
            f"  {emitter:<12} {result.expected_shape(emitter)}  worst extrapolation residual {scatter}"
        )


@app.command()
def plot(
    tabulation: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A tabulation, as written by collect.")
    ],
    emitter: Annotated[str, typer.Option("--emitter", "-e", help="Which component's light.")] = "disk",
    quantity: Annotated[
        str, typer.Option("--quantity", "-q", help="extinction, reddening, or curve.")
    ] = "extinction",
    against: Annotated[
        str, typer.Option("--against", "-a", help="Coordinate along the horizontal axis.")
    ] = "inclination",
    colour_by: Annotated[
        str, typer.Option("--colour-by", "-c", help="Draw a family of curves over this coordinate.")
    ] = "",
    wavelength: Annotated[float, typer.Option(help="Wavelength for the extinction, in microns.")] = 0.55,
    fix: Annotated[
        list[str] | None,
        typer.Option(help="Hold a coordinate at an index, as name=index. Repeatable."),
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the figure.")] = Path(
        "plot.pdf"
    ),
) -> None:
    """Plot a tabulation.

    Covers what the original's nine plotting scripts did, and the combinations
    they did not: any quantity against any axis, at any wavelength.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        typer.echo("plotting needs matplotlib: install this package with its `plots` extra", err=True)
        raise typer.Exit(code=1) from None

    from .plots import plot_extinction, plot_extinction_curve
    from .tabulation import read_tabulation

    held = {}
    for entry in fix or []:
        name, _, index = entry.partition("=")
        if not index.lstrip("-").isdigit():
            typer.echo(f"--fix expects name=index, got {entry!r}", err=True)
            raise typer.Exit(code=1)
        held[name] = int(index)

    table = read_tabulation(str(tabulation))
    if emitter not in table.emitters:
        typer.echo(
            f"no emitter {emitter!r} in this tabulation; it has {', '.join(table.emitters)}",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        if quantity == "curve":
            axes = plot_extinction_curve(table, emitter, colour_by or None, held)
        else:
            axes = plot_extinction(
                table,
                emitter,
                against,
                quantity=quantity,
                wavelength=wavelength,
                colour_by=colour_by or None,
                fixed=held,
            )
    except (KeyError, ValueError) as error:
        # A KeyError's str() carries the repr quotes; its argument does not.
        typer.echo(str(error.args[0]) if error.args else str(error), err=True)
        raise typer.Exit(code=1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    axes.figure.savefig(output)
    plt.close(axes.figure)
    typer.echo(f"wrote {output}")


@app.command()
def compare(
    tabulation: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A tabulation, as written by collect.")
    ],
    atlas: Annotated[
        Path,
        typer.Option(
            "--atlas", "-a", exists=True, dir_okay=False, help="A published atlas to compare against."
        ),
    ],
    tolerance: Annotated[
        float, typer.Option(help="Absolute agreement in transmission to report against.")
    ] = 0.02,
    figure: Annotated[Path | None, typer.Option("--figure", "-f", help="Also draw the two overlaid.")] = None,
    wavelength: Annotated[float, typer.Option(help="Wavelength for the figure, in microns.")] = 0.5512,
) -> None:
    """Compare a tabulation against a published atlas.

    This is what the original's plotting scripts offered as ``--showAtlas``,
    made into a measurement rather than a decoration: it reports how closely the
    two agree, not merely draws them together.

    Note that the Ferrara et al. (1999) atlas is published to two decimal places
    -- some of its entries exceed a transmission of one by exactly the rounding
    -- so agreement closer than about 0.01 cannot be demonstrated against it.
    """
    from .tabulation import read_tabulation
    from .validation import ATLAS_QUANTIZATION, compare_with_atlas, read_atlas

    table = read_tabulation(str(tabulation))
    published = read_atlas(atlas)
    typer.echo(f"{table.label} against {atlas.name}")
    typer.echo(
        f"  the atlas is quantized at {ATLAS_QUANTIZATION:g} in transmission, "
        "which is the floor on any disagreement"
    )
    comparisons = {}
    worst = 0.0
    for emitter in table.emitters:
        if emitter not in published.attenuation:
            continue
        try:
            comparisons[emitter] = compare_with_atlas(table, published, emitter)
        except ValueError as error:
            typer.echo(f"  {emitter}: {error}", err=True)
            raise typer.Exit(code=1) from None
        typer.echo(f"  {comparisons[emitter].summary()}")
        worst = max(worst, comparisons[emitter].median)

    if figure is not None:
        _draw_comparison(table, published, comparisons, figure, wavelength)
        typer.echo(f"  wrote {figure}")
    if worst > tolerance:
        typer.echo(f"median disagreement exceeds {tolerance:g}", err=True)
        raise typer.Exit(code=1)


def _draw_comparison(table, published, comparisons, figure: Path, wavelength: float) -> None:
    """Overlay a tabulation on an atlas, against inclination, at each optical depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = int(np.argmin(np.abs(published.wavelengths - wavelength)))
    fig, panels = plt.subplots(
        1, len(comparisons), figsize=(6.0 * len(comparisons), 4.5), squeeze=False, constrained_layout=True
    )
    for panel, (emitter, comparison) in zip(panels[0], comparisons.items(), strict=True):
        computed = comparison.computed[index]
        reference = comparison.published[index]
        if computed.ndim > 2:
            computed, reference = computed[..., 0], reference[..., 0]
        colours = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, computed.shape[1]))
        for depth in range(computed.shape[1]):
            panel.plot(published.inclinations, computed[:, depth], color=colours[depth], linewidth=2.0)
            panel.plot(
                published.inclinations,
                reference[:, depth],
                color=colours[depth],
                linestyle="none",
                marker="o",
                markersize=4.0,
                markerfacecolor="none",
            )
        panel.set_xlabel(r"Inclination; $i\,[^\circ]$")
        panel.set_ylabel("Transmission")
        panel.set_title(
            f"{emitter} at {published.wavelengths[index]:g} micron\nlines: computed, circles: published",
            fontsize="small",
        )
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=110)
    plt.close(fig)
