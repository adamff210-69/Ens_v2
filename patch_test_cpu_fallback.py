# Fix test_cpu_fallback: it assumed a CPU-only host, but this box has CUDA,
# so t4_dtype() correctly returned fp16 and the assertion failed.
p = "/kaggle/working/tests/smoke_offline.py"
s = open(p).read()
start = s.index("    def test_cpu_fallback(self):")
end = s.index("    def test_turing_is_fp16(self):")
new = '''    def test_cpu_fallback(self):
        # Must mock CPU rather than assume it: on a GPU host (Kaggle with
        # accelerator ON) t4_dtype() correctly returns fp16, which is not a
        # CPU-fallback regression. This test previously failed on any GPU box.
        import torch
        import unittest.mock as mock
        cuda_mod = mock.patch("pipeline.torch.cuda.is_available", return_value=False)
        count_mod = mock.patch("pipeline.torch.cuda.device_count", return_value=0)
        with cuda_mod, count_mod:
            assert P.t4_dtype() == torch.float32
            assert P.VRAMManager.allocate_memory_map() is None
            assert P.gpu_max_memory() is None
            assert P.n_gpus() == 0

    def test_gpu_host_reports_its_own_dtype(self):
        # Mirror of the above: if this session really has CUDA, t4_dtype()
        # must NOT be float32. Guards the mock from masking a real bug.
        import torch
        if not torch.cuda.is_available():
            return                      # CPU box: nothing to assert
        assert P.t4_dtype() in (torch.float16, torch.bfloat16)
        assert P.n_gpus() >= 1

'''
open(p, "w").write(s[:start] + new + s[end:])
print("patched:", p)
