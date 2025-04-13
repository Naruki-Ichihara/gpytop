import cupy as cp

def test_gpu_is_available():
    """
    Test if GPU is available and print device information.
    """
    # Check if GPU is available
    assert cp.cuda.runtime.getDeviceCount() > 0, "No GPU found"
    
    # Get device count
    num_devices = cp.cuda.runtime.getDeviceCount()
    device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    print(f"\nNumber of devices: {num_devices}")
    print(f"Device name: {device_name}")

def test_rapid_is_available():
    """
    Test if RAPIDS is available.
    """
    try:
        import cucim
        assert True, "RAPIDS libraries are available"
    except ImportError as e:
        assert False, f"RAPIDS libraries are not available: {e}"

def test_warp_is_available():
    """
    Test if Warp is available.
    """
    try:
        import warp
        warp.init()
        assert True, "Warp is available"
    except ImportError as e:
        assert False, f"Warp is not available: {e}"

def test_torch_is_available():
    """
    Test if PyTorch is available.
    """
    try:
        import torch
        assert torch.cuda.is_available(), "PyTorch is not available"
    except ImportError as e:
        assert False, f"PyTorch is not available: {e}"

def test_torch_fem_is_available():
    """
    Test if PyTorch FEM is available.
    """
    try:
        import torchfem
        assert True, "Torch-FEM is available"
    except ImportError as e:
        assert False, f"Torch-FEM is not available: {e}"

