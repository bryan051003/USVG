# Mel
num_mels = 80
#text_cleaners = ['english_cleaners']

num_spec = 257
# FastSpeech
vocab_size = 300
pitch_size = 100
max_seq_len = 3000

pitch_encoder_n_layer = 2


encoder_dim = 256
encoder_n_layer = 4
encoder_head = 2
encoder_conv1d_filter_size = 1024

decoder_dim = 256
decoder_n_layer = 4
decoder_head = 2
decoder_conv1d_filter_size = 1024

fft_conv1d_kernel = (3, 1) #(5, 1) (3, 1)
fft_conv1d_padding = (1, 0) #(2, 0) (1, 0)

#duration_predictor_filter_size = 256
#duration_predictor_kernel_size = 3
dropout = 0.1

#SFGAN
sfgan_frequency_range = [[0, 40], [20, 60], [40, 80]]
sfgan_stack_layers = 3
sfgan_d_model = 64
sfgan_kernel_size = 3
sfgan_max_sampling_length = 600

#Ada
singer_latent_size = 256


# Train
checkpoint_path = "./NSinger/model_new"
logger_path = "./NSinger/logger"
#mel_ground_truth = "./mels"
#alignment_path = "./alignments"

batch_size = 8
sv_batch_size = 24
epochs = 100
n_warm_up_step = 500

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 4
