import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_awq(args) -> None:
    from inference.quantization import quantize_awq
    quantize_awq(args.model, args.output, calib_data=args.calib_data)


def cmd_gptq(args) -> None:
    from inference.quantization import quantize_gptq
    quantize_gptq(
        args.model, args.output,
        bits=args.bits,
        group_size=args.group_size,
        calib_data=args.calib_data,
    )


def cmd_bnb(args) -> None:
    from inference.quantization import quantize_bitsandbytes
    quantize_bitsandbytes(args.model, args.output, method=args.method)


def main():
    p = argparse.ArgumentParser(description="Offline LLM quantization")
    sub = p.add_subparsers(dest="cmd", required=True)

    # AWQ
    awq = sub.add_parser("awq", help="4-bit AWQ quantization (recommended)")
    awq.add_argument("--model", required=True)
    awq.add_argument("--output", required=True)
    awq.add_argument("--calib-data", default="pileval", dest="calib_data")

    # GPTQ
    gptq = sub.add_parser("gptq", help="4-bit GPTQ quantization")
    gptq.add_argument("--model", required=True)
    gptq.add_argument("--output", required=True)
    gptq.add_argument("--bits", type=int, default=4, choices=[3, 4])
    gptq.add_argument("--group-size", type=int, default=128, dest="group_size")
    gptq.add_argument("--calib-data", default="wikitext2", dest="calib_data")

    # BnB
    bnb = sub.add_parser("bnb", help="bitsandbytes quantization (runtime only)")
    bnb.add_argument("--model", required=True)
    bnb.add_argument("--output", required=True)
    bnb.add_argument("--method", default="4bit", choices=["4bit", "8bit"])

    args = p.parse_args()

    from utils.logging import setup_logging
    setup_logging({"level": "INFO"})

    dispatch = {"awq": cmd_awq, "gptq": cmd_gptq, "bnb": cmd_bnb}
    dispatch[args.cmd](args)

    print(f"\nDone. Artifact saved to: {args.output}")
    print(f"Next: python scripts/upload_s3.py --path {args.output} --bucket <bucket> --prefix <prefix>")


if __name__ == "__main__":
    main()
