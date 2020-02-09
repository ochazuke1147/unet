from src.func_processing import *


video_paths = ['./datasets/movie/hayashi_n4.avi', './datasets/movie/kikuchi_n2.avi',
               './datasets/movie/kurose_n1.avi', './datasets/movie/okazawa8.avi']

fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # fourccを定義
out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (1024, 768), isColor=False)  # 動画書込準備

cap_user = cv2.VideoCapture(video_paths[3])
cap_user.set(cv2.CAP_PROP_POS_FRAMES, 0)

input_channel_count = 1
output_channel_count = 1
BATCH_SIZE = 1
mode = 'U-Net'
if mode == 'U-Net':
    print('U-Net mode.')
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights('unet_weights.hdf5')
elif mode == 'SegNet':
    print('SegNet mode.')
    first_layer_filter_count = 64
    model = segnet(input_channel_count, output_channel_count, first_layer_filter_count)
    model.load_weights('segnet_weights.hdf5')
else:
    print('modelが不正です.')
    exit(1)

ret, image_user = cap_user.read()

while ret:
    image_user_gray = cv2.cvtColor(image_user, cv2.COLOR_BGR2GRAY)
    size = (image_user_gray.shape[1], image_user_gray.shape[0])
    images = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    image = cv2.resize(image_user_gray, (IMAGE_SIZE, IMAGE_SIZE))
    image = image[:, :, np.newaxis]
    images[0] = normalize_x(image)
    Y_pred = model.predict(images, BATCH_SIZE)
    y = cv2.resize(Y_pred[0], size)
    y_dn = denormalize_y(y)
    y_dn = np.uint8(y_dn)
    _, mask = cv2.threshold(y_dn, 0, 255, cv2.THRESH_OTSU)

    masked = cv2.bitwise_and(image_user_gray, mask)
    mask_rest = cv2.bitwise_not(mask)
    masked = cv2.bitwise_or(masked, mask_rest)

    print(masked.dtype)
    cv2.imshow('', masked)
    cv2.waitKey()

    out.write(masked)
    ret, image_user = cap_user.read()

out.release()
