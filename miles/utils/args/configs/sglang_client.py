from miles.utils.args.schema import A, Arg, BaseConfig


class SglangClientConfig(BaseConfig):
    sglang_server_concurrency: A[int, Arg()] = 512
    eval_server_concurrency: A[
        int | None,
        Arg(
            help=(
                "Requests in flight per dedicated eval engine (--eval-num-gpus), in place of "
                "--sglang-server-concurrency, which it defaults to. Sizes the eval rollout's "
                "concurrency and its share of the HTTP client pool, so a small eval pool can keep as "
                "many episodes going as the training engines do."
            )
        ),
    ] = None
