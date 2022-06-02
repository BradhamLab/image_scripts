"""Script to z-normalize images"""
import argparse
import os

from aicsimageio import AICSImage
from intensipy import models
from skimage import io, exposure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize intensities in 3D images using the Intensify3D algorithm"
    )
    parser.add_argument(
        "image",
        metavar="image",
        type=str,
        help="Image to normalize. Should be a file format that can be read by `aicsimageio`.",
    )
    parser.add_argument(
        "channels",
        metavar="channels",
        type=int,
        nargs="+",
        help="Channels to normalize. Should be passed as a series of zero-indexed integers.",
    )
    parser.add_argument(
        "--fmt",
        dest="format",
        metavar="format",
        type=str,
        default="3D",
        choices=["2d", "3d", "2D", "3D"],
        help="Whether to output 3D tensors or 2D max projections of the normalized image.",
    )

    parser.add_argument(
        "--out",
        dest="out",
        metavar="out",
        type=str,
        default="",
        help="Output file to write. If none, the file name will be generated programmatically from the input file.",
    )
    parser.add_argument(
        "--z_start",
        dest="z_start",
        metavar="z_start",
        type=int,
        default=0,
        help="Z index where signal starts. Should be 0 indexed. Default is 0.",
    )
    parser.add_argument(
        "--z_stop",
        dest="z_stop",
        metavar="z_stop",
        type=int,
        default=-1,
        help="Z index where signal ends. Should be 0 indexed. Default is the last slice.",
    )

    parser.add_argument(
        "--verbose",
        dest="verbose",
        metavar="verbose",
        type=bool,
        default=False,
        help="Whether to print verbosely. Default is False.",
    )

    norm_args = parser.add_argument_group("intensipy arguments")
    norm_args.add_argument(
        "--xy_norm",
        dest="xy_norm",
        metavar="xy_norm",
        type=bool,
        default=False,
        help="Whether to normalize image slices along the XY plane.",
    )
    norm_args.add_argument(
        "--z_norm",
        dest="z_norm",
        metavar="z_norm",
        type=bool,
        default=True,
        help="Whether to normalize image slices along the Z plane.",
    )
    norm_args.add_argument(
        "--dy",
        dest="dy",
        metavar="dy",
        type=int,
        default=29,
        help="Y-diameter of expected object shape. Used for smoothing. Default is 29",
    )
    norm_args.add_argument(
        "--dx",
        dest="dx",
        metavar="dx",
        type=int,
        default=29,
        help="X-diameter of expected object shape. Used for smoothing. Default is 29",
    )
    norm_args.add_argument(
        "--quantiles",
        dest="quantiles",
        metavar="quantiles",
        type=int,
        default=10000,
        help="Number of quantiles to calculate during semi-quantile normalization. Default is 10,000",
    )
    norm_args.add_argument(
        "--smooth_quartiles",
        dest="smooth_quartiles",
        metavar="smooth_quartiles",
        type=bool,
        default=True,
        help="Whether to smooth quantile normalized pixels. Default is True.",
    )
    norm_args.add_argument(
        "--original_scale",
        dest="original_scale",
        metavar="original_scale",
        type=bool,
        default=True,
        help="Whether minimum and maximum values should be consistent with the original image bit depth. Default is True.",
    )
    norm_args.add_argument(
        "--stretch",
        dest="stretch",
        metavar="stretch",
        type=str,
        default="skimage",
        choices=["skimage", "intensify3d"],
        help="Method used to perform contrast stretching. Default is 'skimage'.",
    )

    args = parser.parse_args()
    img = AICSImage(args.image)
    for i in args.channels:
        data = exposure.rescale_intensity(
            img.get_image_data("ZYX", C=i)[args.z_start : args.z_stop],
            in_range=(0, 2 ** img.metadata["attributes"].bitsPerComponentSignificant),
            out_range="dtype",
        )
        norm_model = models.Intensify(
            xy_norm=args.xy_norm,
            z_norm=args.z_norm,
            dy=args.dy,
            dx=args.dx,
            n_quantiles=args.quantiles,
            smooth_quartiles=args.smooth_quartiles,
            keep_original_scale=args.original_scale,
            stretch_method=args.stretch,
            bits=16,
        )
        normed = norm_model.normalize(data.astype(float), verbose=args.verbose)
        if args.format.lower() == "2d":
            normed = normed.max(axis=0)
        if args.out == "":
            out_fn = os.path.join(
                os.path.dirname(args.image),
                os.path.basename(os.path.splitext(args.image)[0])
                + f"_normed_c={i}.tiff",
            )
        elif len(args.channels) > 1:
            ext_split = os.path.splitext(args.out)
            out_fn = ext_split[0] + f"_c={i}{ext_split[1]}"
        else:
            out_fn = args.out
        io.imsave(out_fn, normed)
