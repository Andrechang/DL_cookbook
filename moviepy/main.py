import os
import tarfile
from six.moves import urllib
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#post process alpha
def post_alpha(alpha):
    alpha = (255 * alpha).astype(np.uint8) #int
    # alpha = posemodel.pose_segment(img, alpha)
    contours, hierarchy = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 100:#remove small contour areas
            continue
        cv2.fillPoly(alpha, pts=[contours[i]], color=(255))
    alpha = alpha.astype(np.float32) / 255 #float
    alpha = cv2.erode(alpha, kernel_er, iterations=1)
    alpha = cv2.dilate(alpha, kernel_dil, iterations=1)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (31, 31), 0)
    alpha = (255 * alpha).astype(np.uint8) #int
    return alpha

#resize images
def resize_canvas(input_img, bgr_img):
    if len(input_img.shape) == 2:
        img = np.zeros((bgr_img.shape[0], bgr_img.shape[1]))
        img[0:input_img.shape[0], 0:input_img.shape[1]] = input_img
    else:
        img = np.zeros((bgr_img.shape[0], bgr_img.shape[1], 3))
        img[0:input_img.shape[0], 0:input_img.shape[1], :] = input_img
    return img

#get moviepy time format
def duration_time(dur):
    hour = int(dur//3600)
    dur = dur%3600
    minu = int(dur//60)
    dur = dur%60
    sec = int(dur)
    return (hour, minu, sec)

#combine foreground, alpha and background
def composite4(fg, bg, a):
    fg = np.array(fg, np.float32)
    alpha = np.expand_dims(a / 255,axis=2)
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im

#instantiate deeplab model and run inference
class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        # """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        #Runs inference on a single image.
        #Args: image numpy array
        #Returns:
        #resized_image: RGB image resized from original input image.
        #seg_map: Segmentation map of `resized_image`.
        width, height = image.shape[1], image.shape[0]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

# setup AI
MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

model_dir = 'deeplab_model'
if not os.path.exists(model_dir):
  tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
if not os.path.exists(download_path):
  print('downloading model to %s, this might take a while...' % download_path)
  urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                             download_path)
  print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

#------------------------------------------------------------------------------
import youtube_dl
from moviepy.editor import *
def download_youtube(path, outname, music=False):
    ydl_opts = {'outtmpl': outname,'cachedir': False}
    if music == True:
        ydl_opts['format'] = 'bestaudio/best'
        ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}]
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([path])

download_youtube('https://www.youtube.com/watch?v=Grr15bMzHhY', 'input')
download_youtube('https://www.youtube.com/watch?v=5UZz3EKqPsk', 'back')
download_youtube('https://www.youtube.com/watch?v=YJQqwf4L9HE', 'music.mp3', music=True)

clipp = VideoFileClip('input.mp4').subclip((0,10), (1,10))
clipp.write_videofile('input2.mp4', fps=30)

parser = argparse.ArgumentParser(description='Segmentation')
parser.add_argument('-i', '--input_dir', type=str, default='input2.mp4', help='input to do segmentation')
parser.add_argument('-b', '--background', type=str, default='back.mkv', help='background to overlay')
parser.add_argument('-m', '--music', type=str, default='music.mp3', help='music')
parser.add_argument('-o', '--output', type=str, default='output.mp4', help='output file')
args=parser.parse_args()

#BG
path_bg = args.background
print('background: ', path_bg)
bgr_isimg = False
bext = os.path.splitext(path_bg)[1]
if bext == '.png' or bext == '.jpg' or bext == '.jpeg':
    bgr_isimg = True
    bgr_img = cv2.imread(path_bg)
    bgr_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
else:
    bgr_cap = cv2.VideoCapture(path_bg)
    bgr_fps = bgr_cap.get(cv2.CAP_PROP_FPS)
    bgr_length = int(bgr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, bgr_img = bgr_cap.read()

screen_size = (1920, 1080)
screen_w = int(screen_size[0])
screen_h = int(screen_size[1])

#VIDEO
path_vid = args.input_dir
print('input vid: ', path_vid)
cap = cv2.VideoCapture(path_vid)
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#AUDIO
path_audio = args.music
print('audio: ', path_audio)

#OUT
outname = args.output
print('output: ', outname)
tmp_outname = 'tmp_output.avi'
out = cv2.VideoWriter(tmp_outname, cv2.VideoWriter_fourcc(*'DIVX'), fps, screen_size)

pbar = tqdm(total = length)
audio_i = 0#frame index for audio bar
bgr_i = 0#frame index for bgr video

while(cap.isOpened()):
    pbar.update(1)
    ret, img = cap.read()
    if ret == False:
        break
    if screen_w < img.shape[1] or screen_h < img.shape[0]:
        img = cv2.resize(img, screen_size)
#AI: run segmentation
    res_im, seg = MODEL.run(img)
    seg=cv2.resize(seg.astype(np.uint8), (img.shape[1], img.shape[0]))
    mask_sel=(seg==15).astype(np.float32)
#MASK: post-process segmentation mask
    alpha = mask_sel.astype(np.float32)
    alpha[alpha > 0.2] = 1
    alpha_out0 = post_alpha(alpha)
    # cv2.imwrite('alpha.png', alpha_out0)


#FRAME: Combine alpha, image and background
    if not bgr_isimg:
        bgret, bgr_img = bgr_cap.read()
        if bgret == False or bgr_i >= bgr_length:
            bgr_i = 0
            bgr_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, bgr_img = bgr_cap.read()
            print('loop bgr video')
        bgr_img = cv2.resize(bgr_img, screen_size)
        bgr_i += 1

    fg_img = resize_canvas(img, bgr_img)
    alpha = resize_canvas(alpha_out0, bgr_img)
    comp_im_tr1 = bgr_img
    comp_im_tr1 = composite4(fg_img, comp_im_tr1, alpha)

    outt = comp_im_tr1
    out.write(outt)

out.release()
print('Segmentation Done')

clipp = VideoFileClip(tmp_outname)
audio = AudioFileClip(path_audio).subclip((0,0), duration_time(clipp.duration))
clipp.audio = audio
clipp.write_videofile(outname, fps=60)
print('Done')



