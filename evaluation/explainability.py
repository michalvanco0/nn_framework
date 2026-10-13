import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return

        raise ValueError(
            f"Target layer '{self.target_layer}' was not found."
        )

    def generate_heatmap(self, input_image, class_index=None):
        self.model.eval()

        output = self.model(input_image)

        if class_index is None:
            class_index = output.argmax(dim=1).item()

        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, class_index] = 1

        output.backward(gradient=one_hot)

        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.sum(
            weights * self.activations,
            dim=1,
        )

        cam = F.relu(cam)
        cam = cam[0]

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()