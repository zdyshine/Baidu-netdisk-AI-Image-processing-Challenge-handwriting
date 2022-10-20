import argparse
from PIL import Image
import torch
import torchvision.transforms as T
import onnx
import os
from torchviz import make_dot
import onnxruntime as ort
import numpy as np
import copy

parser = argparse.ArgumentParser(description='Test inpainting')
parser.add_argument("--image", type=str,
                    default="examples/inpaint/case1.png", help="path to the image file")
parser.add_argument("--mask", type=str,
                    default="examples/inpaint/case1_mask.png", help="path to the mask file")
parser.add_argument("--out", type=str,
                    default="examples/inpaint/case1_out_test.png", help="path for the output file")
parser.add_argument("--checkpoint", type=str,
                    default="pretrained/states_tf_places2.pth", help="path to the checkpoint file")


def main():

    args = parser.parse_args()

    generator_state_dict = torch.load(args.checkpoint)['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks_test import Generator
    else:
        from model.networks_tf import Generator  

    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(args.checkpoint)['G']
    generator.load_state_dict(generator_state_dict, strict=True)
    generator.eval()
    
    print('Converting checkpoint to onnx format')
    ckpt_name = os.path.basename(args.checkpoint).split('.')[0]
    ckpt_dir = os.path.dirname(args.checkpoint)

    # load image and mask
    image = Image.open(args.image)
    mask = Image.open(args.mask)

    # prepare input
    image = T.ToTensor()(image).float().to(device)
    mask = T.ToTensor()(mask).float().to(device)

    _, h, w = image.shape
    grid = 8
    test_size = 512
    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    image = torch.nn.functional.interpolate(image, size=(test_size,test_size))
    mask = torch.nn.functional.interpolate(mask, size=(test_size,test_size))
    print(f"Shape of image: {image.shape}")

    image = (image*2 - 1.)  # map image values to [-1, 1] range
    mask = (mask > 0.5).float()  # 1.: masked 0.: unmasked

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :].to(device)
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels
    # x = x.repeat(8, 1, 1, 1)
    # mask = mask.repeat(8, 1, 1, 1)
    # with torch.inference_mode():
    #     _, x_stage2 = generator(x, mask)
    
    
    # complete image
    
    # image_inpainted = image * (1.-mask) + x_stage2 * mask
    # save inpainted image
    
    # img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    # img_out = img_out.to(device='cpu', dtype=torch.uint8)
    # img_out = Image.fromarray(img_out.numpy())
    # img_out.save(args.out)

    print(f"Saved output file at: {args.out}")
    
    torch.onnx.export(
        generator,
        (x, mask),
        os.path.join(ckpt_dir, ckpt_name+'.onnx'),
        verbose=True,
        export_params=True,
        input_names=['img', 'mask'],
        output_names=['output_stage1', 'output_stage2'],
        do_constant_folding=True,
        opset_version=11)
        # dynamic_axes={'img':[0],
        #               'mask':[0],
        #               'output_stage1':[0],
        #               'output_stage2':[0]})
    
    print('testing onnx model...')
    model = onnx.load(os.path.join(ckpt_dir, ckpt_name+'.onnx'))
    onnx.checker.check_model(model)

    session = ort.InferenceSession(os.path.join(ckpt_dir, ckpt_name+'.onnx'))
    x = x.cpu().numpy().astype(np.float32)  # 注意输入type一定要np.float32
    mask = mask.cpu().numpy().astype(np.float32)
    out_stage1, out_stage2 = session.run(None, { 'img': x, 'mask': mask })
    print('onnx model test finished')
    print(out_stage1.shape)
    print(out_stage2.shape)
    
    with torch.inference_mode():
        x_stage1, x_stage2 = generator(torch.from_numpy(x).to(device).float(), torch.from_numpy(mask).to(device).float())
        # np.testing.assert_allclose(x_stage2.cpu().numpy()[0], out_stage2[0], rtol=1e-03, atol=1e-05)
        # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        
        ort_inpainted = image.cpu().numpy() * (1.-mask) + out_stage2 * mask
        # save inpainted image
        img_out = ((ort_inpainted[0].transpose(1, 2, 0) + 1)*127.5)
        img_out = img_out.astype(np.uint8)
        img_out = Image.fromarray(img_out)
        img_out.save(args.out+'.onnx.png')

        print(f"Saved ort output file at: {args.out}")


if __name__ == '__main__':
    main()
