import unittest

TORCH_STACK_AVAILABLE = True

try:
    import torch
    from models.resnet import create_resnet18
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    TORCH_STACK_AVAILABLE = False


class TestResnetModel(unittest.TestCase):
    @unittest.skipUnless(TORCH_STACK_AVAILABLE, "PyTorch/torchvision is not installed")
    def test_create_resnet18_outputs_correct_shape(self) -> None:
        model = create_resnet18(num_classes=10)
        model.eval()

        batch = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(batch)

        self.assertEqual(output.shape, (2, 10))
