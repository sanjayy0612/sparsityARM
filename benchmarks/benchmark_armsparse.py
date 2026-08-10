"""Reference contiguous block-sparse FFN microbenchmark."""
import time, json
from common import base_parser, write_result


def main() -> None:
    parser=base_parser(__doc__); parser.add_argument("--sparsity",type=float,required=True); parser.add_argument("--block-size",type=int,required=True,choices=[8,16,32,64,128]); parser.add_argument("--hidden-size",type=int,default=1024); parser.add_argument("--intermediate-size",type=int,default=4096)
    args=parser.parse_args()
    import torch
    from armsparse.kernels import PackedFFNWeights, block_sparse_ffn
    from armsparse.sparsity import blockify
    torch.manual_seed(args.seed); hidden=torch.randn(1,args.hidden_size); gate=torch.randn(args.intermediate_size,args.hidden_size); up=torch.randn_like(gate); down=torch.randn(args.hidden_size,args.intermediate_size)
    block=blockify(torch.nn.functional.linear(hidden,gate).abs()[0],args.block_size,args.sparsity); packed=PackedFFNWeights.from_weights(gate,up,down,args.block_size)
    for _ in range(args.warmup): block_sparse_ffn(hidden,packed,block.values)
    times=[]
    for _ in range(args.repeats):
        t=time.perf_counter(); output=block_sparse_ffn(hidden,packed,block.values); times.append((time.perf_counter()-t)*1000)
    record={"mode":"armsparse","model":args.model,"reference_microbenchmark":True,"target_sparsity":args.sparsity,"realized_sparsity":block.realized_sparsity,"block_size":args.block_size,"active_blocks":block.active_blocks,"ffn_ms_median":sorted(times)[len(times)//2],"output_finite":bool(torch.isfinite(output).all()),"threads":args.threads}
    path=write_result(record,args.output_dir); print(json.dumps(record,indent=2)); print(f"Saved {path}")

if __name__ == "__main__": main()
