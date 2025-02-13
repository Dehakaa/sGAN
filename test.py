"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: The wandb package is missing. "--use_wandb" will cause an error.')

def setup_test_environment():
    # 获取测试配置并调整某些参数
    options = TestOptions().parse()
    options.num_threads = 0           # 仅支持 num_threads = 0
    options.batch_size = 1            # 仅支持 batch_size = 1
    options.serial_batches = True     # 禁用数据洗牌
    options.no_flip = True            # 禁止图片翻转
    options.display_id = -1           # 禁用 visdom 显示，结果会保存为 HTML 文件
    return options

def initialize_logger(options):
    # 如果启用 wandb，则初始化它
    if options.use_wandb:
        wandb_run = wandb.init(project=options.wandb_project_name, name=options.name, config=options) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

def create_webpage_directory(options):
    # 创建网页保存目录
    web_dir = os.path.join(options.results_dir, options.name, f'{options.phase}_{options.epoch}')
    if options.load_iter > 0:
        web_dir = f'{web_dir}_iter{options.load_iter}'
    print(f'Creating webpage directory: {web_dir}')
    return html.HTML(web_dir, f'Experiment = {options.name}, Phase = {options.phase}, Epoch = {options.epoch}')

def test_model_on_dataset(model, dataset, options, webpage):
    # 在数据集上运行测试，并将结果保存到网页
    for idx, data in enumerate(dataset):
        if idx >= options.num_test:
            break
        model.set_input(data)          # 解包数据
        model.test()                   # 进行推理
        visuals = model.get_current_visuals()  # 获取结果图像
        img_paths = model.get_image_paths()    # 获取图像路径
        if idx % 5 == 0:  # 每隔 5 张保存一次图像
            print(f'Processing {idx:04d}-th image... {img_paths}')
        save_images(webpage, visuals, img_paths, aspect_ratio=options.aspect_ratio, width=options.display_winsize, use_wandb=options.use_wandb)

def main():
    options = setup_test_environment()  # 设置测试环境
    dataset = create_dataset(options)  # 创建数据集
    model = create_model(options)      # 创建模型
    model.setup(options)               # 设置模型

    # 初始化日志记录
    initialize_logger(options)

    # 创建网页目录
    webpage = create_webpage_directory(options)

    # 设置模型为评估模式（如果需要）
    if options.eval:
        model.eval()

    # 在数据集上测试模型并保存结果
    test_model_on_dataset(model, dataset, options, webpage)

    # 保存网页
    webpage.save()

if __name__ == '__main__':
    main()

