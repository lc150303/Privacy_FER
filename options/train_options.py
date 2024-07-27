from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
        self._parser.add_argument('--model', type=str, default='LRPPN', help='model to run')

        self._parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=600, help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=300, help='frequency of saving the latest results')

        self._parser.add_argument('--save_img', action='store_true', help='whether save fake imgs')
        self._parser.add_argument('--save_fake_dir', type=str, default='generate', help='dir to save fake image')
        self._parser.add_argument('--save_results_file', type=str, default='results.csv', help='dir to save results during training')

        self._parser.add_argument('--pretrain_nepochs', type=int, default=10, help='# of epochs to pretrain')
        self._parser.add_argument('--nepochs_no_decay', type=int, default=20, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')

        self._parser.add_argument('--pretrain', action='store_true', help='enable pretrain')
        self._parser.add_argument('--save_model', action='store_true', help='save neural model')
        self._parser.add_argument('--save_model_freq', type=int, default=1, help='frequency of saving model')
        self._parser.add_argument('--save_features', type=int, default=1, help='save low-resolution representations')

        self._parser.add_argument('--lr_En', type=float, default=0.0001, help='initial learning rate for En adam')
        self._parser.add_argument('--En_adam_b1', type=float, default=0.5, help='beta1 for En adam')
        self._parser.add_argument('--En_adam_b2', type=float, default=0.999, help='beta2 for En adam')

        self._parser.add_argument('--lr_C', type=float, default=0.0001, help='initial learning rate for C_id\C_e adam')
        self._parser.add_argument('--C_adam_b1', type=float, default=0.5, help='beta1 for C_id\C_e adam')
        self._parser.add_argument('--C_adam_b2', type=float, default=0.999, help='beta2 for C_id\C_e adam')

        self._parser.add_argument('--lr_De', type=float, default=0.0001, help='initial learning rate for De adam')
        self._parser.add_argument('--De_adam_b1', type=float, default=0.5, help='beta1 for De adam')
        self._parser.add_argument('--De_adam_b2', type=float, default=0.999, help='beta2 for De adam')

        self._parser.add_argument('--L_cross', type=float, default=0.5, help='adversarial loss weight')
        self._parser.add_argument('--L_adv', type=float, default=0.5, help='adversarial loss weight')

        self._parser.add_argument('--L_cls_sim', type=float, default=0.3, help='classification similarity loss weight')
        self._parser.add_argument('--L_lir', type=float, default=0.3, help='loss inequality regulation weight')

        self._parser.add_argument('--L_cons_sim', type=float, default=8, help='recontruction similarity loss weight')
        self._parser.add_argument('--L_cyc', type=float, default=5, help='recontruction loss weight')

        self.is_train = True
