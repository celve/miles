import importlib

_PATCHED = "_miles_fused_adam_batches"
# TE allocates NVTETensor handles from a fixed pool of 20,321 per multi-tensor call.
MAX_HANDLES = 16384


def install_fused_adam_batches_patch(fused_adam=None) -> None:
    """Split TE FusedAdam's multi-tensor calls so one never needs more tensor handles than TE can allocate.

    A Qwen3.5-35B-A3B rank at EP 4 holds ~5K parameter tensors (every expert is its own weight), and with master
    weights one Adam call takes 5 lists of them, which fails with "Maximum number of tensors reached: 20321".
    """
    if fused_adam is None:
        try:
            fused_adam = importlib.import_module("transformer_engine.pytorch.optimizers.fused_adam")
        except ImportError:
            return
    applier = getattr(fused_adam, "multi_tensor_applier", None)
    if applier is None or getattr(applier, _PATCHED, False):
        return

    def batched(op, noop_flag, tensor_lists, *args):
        per_call = max(1, MAX_HANDLES // max(1, len(tensor_lists)))
        count = len(tensor_lists[0]) if tensor_lists else 0
        if count <= per_call:
            return applier(op, noop_flag, tensor_lists, *args)
        for start in range(0, count, per_call):
            applier(op, noop_flag, [ts[start : start + per_call] for ts in tensor_lists], *args)

    setattr(batched, _PATCHED, True)
    fused_adam.multi_tensor_applier = batched
