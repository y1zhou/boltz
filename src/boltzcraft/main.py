"""Entrypoint for using Boltz-1 as a design model."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    BoltzDiffusionParams,
    BoltzProcessedInput,
    check_inputs,
    download,
    process_inputs,
)
from boltz.model.model import Boltz1


def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Run predictions with Boltz-1.

    @click.option(
        "--out_dir",
        type=click.Path(exists=False),
        help="The path where to save the predictions.",
        default="./",
    )
    @click.option(
        "--cache",
        type=click.Path(exists=False),
        help="The directory where to download the data and model. Default is ~/.boltz.",
        default="~/.boltz",
    )
    @click.option(
        "--checkpoint",
        type=click.Path(exists=True),
        help="An optional checkpoint, will use the provided Boltz-1 model by default.",
        default=None,
    )
    @click.option(
        "--devices",
        type=int,
        help="The number of devices to use for prediction. Default is 1.",
        default=1,
    )
    @click.option(
        "--accelerator",
        type=click.Choice(["gpu", "cpu", "tpu"]),
        help="The accelerator to use for prediction. Default is gpu.",
        default="gpu",
    )
    @click.option(
        "--recycling_steps",
        type=int,
        help="The number of recycling steps to use for prediction. Default is 3.",
        default=3,
    )
    @click.option(
        "--sampling_steps",
        type=int,
        help="The number of sampling steps to use for prediction. Default is 200.",
        default=200,
    )
    @click.option(
        "--diffusion_samples",
        type=int,
        help="The number of diffusion samples to use for prediction. Default is 1.",
        default=1,
    )
    @click.option(
        "--step_scale",
        type=float,
        help="The step size is related to the temperature at which the diffusion process samples the distribution."
        "The lower the higher the diversity among samples (recommended between 1 and 2). Default is 1.638.",
        default=1.638,
    )
    @click.option(
        "--write_full_pae",
        type=bool,
        is_flag=True,
        help="Whether to dump the pae into a npz file. Default is True.",
    )
    @click.option(
        "--write_full_pde",
        type=bool,
        is_flag=True,
        help="Whether to dump the pde into a npz file. Default is False.",
    )
    @click.option(
        "--output_format",
        type=click.Choice(["pdb", "mmcif"]),
        help="The output format to use for the predictions. Default is mmcif.",
        default="mmcif",
    )
    @click.option(
        "--num_workers",
        type=int,
        help="The number of dataloader workers to use for prediction. Default is 2.",
        default=2,
    )
    @click.option(
        "--override",
        is_flag=True,
        help="Whether to override existing found predictions. Default is False.",
    )
    @click.option(
        "--seed",
        type=int,
        help="Seed to use for random number generator. Default is None (no seeding).",
        default=None,
    )
    @click.option(
        "--use_msa_server",
        is_flag=True,
        help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
    )
    @click.option(
        "--msa_server_url",
        type=str,
        help="MSA server url. Used only if --use_msa_server is set. ",
        default="https://api.colabfold.com",
    )
    @click.option(
        "--msa_pairing_strategy",
        type=str,
        help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
        default="greedy",
    )
    """
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache_path = Path(cache).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data_path = Path(data).expanduser()
    out_dir_path = Path(out_dir).expanduser()
    out_dir_path = out_dir_path / f"boltz_results_{data_path.stem}"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache_path)

    # Validate inputs
    data_paths = check_inputs(data_path, out_dir_path, override)
    if not data_paths:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data_paths) < devices:
            msg = (
                "Number of requested devices is greater than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data_paths)} structure"
    msg += "s" if len(data_paths) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache_path / "ccd.pkl"
    process_inputs(
        data=data_paths,
        out_dir=out_dir_path,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir_path / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint_path = cache_path / "boltz1_conf.ckpt"
    else:
        checkpoint_path = Path(checkpoint).expanduser()

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        predict_args=predict_args,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=str(processed.targets_dir),
        output_dir=str(out_dir_path / "predictions"),
        output_format=output_format,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )
