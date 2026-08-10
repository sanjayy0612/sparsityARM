"""Reference generic sparse FFN microbenchmark. This does not yet patch model generation."""
import time, json
from common import base_parser, write_result


def main() -> None:
    parser = base_parser(__doc__); parser.add_argument("--sparsity", type=float, required=True); parser.add_argument("--hidden-size",type=int,default=1024); parser.add_argument("--intermediate-size",type=int,default=4096)
    args = parser.parse_args()
    import torch
    from armsparse.kernels import sparse_ffn
    from armsparse.sparsity import activation_scores, make_mask
    torch.manual_seed(args.seed); hidden=torch.randn(1,args.hidden_size); gate=torch.randn(args.intermediate_size,args.hidden_size); up=torch.randn_like(gate); down=torch.randn(args.hidden_size,args.intermediate_size)
    scores=activation_scores(torch.nn.functional.linear(hidden,gate)); mask=make_mask(scores,args.sparsity)
    for _ in range(args.warmup): sparse_ffn(hidden,gate,up,down,mask.values)
    times=[]
    for _ in range(args.repeats):
        t=time.perf_counter(); output=sparse_ffn(hidden,gate,up,down,mask.values); times.append((time.perf_counter()-t)*1000)
    record={"mode":"sparse","model":args.model,"reference_microbenchmark":True,"sparsity":args.sparsity,"realized_sparsity":mask.realized_sparsity,"active_neurons":mask.active_count,"ffn_ms_median":sorted(times)[len(times)//2],"output_finite":bool(torch.isfinite(output).all()),"threads":args.threads}
    path=write_result(record,args.output_dir); print(json.dumps(record,indent=2)); print(f"Saved {path}")

if __name__ == "__main__": main()
