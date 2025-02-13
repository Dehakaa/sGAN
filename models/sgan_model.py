import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class SGANModel(BaseModel):
    """
    This class implements the SGAN model, which incorporates a Self-Attention mechanism
    (Vaswani et al., 2017) into the CycleGAN architecture to improve streak recognition.

    SGAN extends the CycleGAN model by integrating attention modules in the generators.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options and rewrite default values for existing options."""
        # Set default values specific to SGAN
        parser.set_defaults(no_dropout=True)  # Default SGAN does not use dropout
        if is_train:
            # Define weights for the cycle loss terms
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # Weight for identity mapping loss
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping loss')

        return parser

    def __init__(self, opt):
        """Initialize the SGAN class."""
        # Call the base class constructor
        BaseModel.__init__(self, opt)

        # Define the loss names to track
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # Specify which images should be visualized or saved
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # Include identity images if applicable
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # Combine both visual name lists
        self.visual_names = visual_names_A + visual_names_B

        # Define which models to save to disk
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # Define the generators (with self-attention mechanism)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'resnet_9blocks_with_attention',
                                        opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'resnet_9blocks_with_attention',
                                        opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Define the discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Assert that identity loss is only used when input and output channels are equal
            if opt.lambda_identity > 0.0:
                assert (opt.input_nc == opt.output_nc)
            # Create image pools for fake A and B images during training
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # Initialize optimizers for both generators and discriminators
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # Add optimizers to the list for easy access during training
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        AtoB = self.opt.direction == 'AtoB'  # Check if the direction is A to B or B to A
        # Set real images for A and B based on the direction
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # Store image paths for visualization
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass."""
        # Generate fake images using the respective generators
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator."""
        # Compute discriminator's prediction for real and fake images
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)  # Loss for real images
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)  # Loss for fake images
        loss_D = (loss_D_real + loss_D_fake) * 0.5  # Average the losses
        loss_D.backward()  # Backpropagate the discriminator's loss
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A."""
        fake_B = self.fake_B_pool.query(self.fake_B)  # Get a fake image from the pool
        # Compute the loss for discriminator D_A
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B."""
        fake_A = self.fake_A_pool.query(self.fake_A)  # Get a fake image from the pool
        # Compute the loss for discriminator D_B
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B."""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            # If identity loss is used, compute the loss for G_A and G_B
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Compute GAN loss for generators
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Compute cycle consistency loss for both directions
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Total generator loss is the sum of GAN and cycle consistency losses
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()  # Backpropagate the generator's loss

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights."""
        self.forward()  # Run forward pass
        # Update the generators first (freeze the discriminators)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()  # Update the generator weights
        # Now update the discriminators (freeze the generators)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # Update discriminator D_A
        self.backward_D_B()  # Update discriminator D_B
        self.optimizer_D.step()  # Update the discriminator weights
