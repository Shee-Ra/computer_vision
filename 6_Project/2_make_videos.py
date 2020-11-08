import cv2
import os

# paths etc.
path_left_images = os.path.join(os.getcwd(),'6_Project/data/processed/')
path_depth_maps = os.path.join(os.getcwd(),'6_Project/data/results/')
path_save_videos = os.path.join(os.getcwd(),'6_Project/data/results/')

left_images = sorted(os.listdir(path_left_images))
depth_maps = sorted(os.listdir(path_depth_maps))

video_left_images = 'video_left_images.avi'
video_depth_maps = 'video_depth_maps.avi'

left = [path_left_images, left_images, video_left_images, path_save_videos]
depth_maps = [path_depth_maps, depth_maps, video_left_images, path_save_videos]

# create videos
for im_sets in [left, depth_maps]:
    images = [img for img in im_sets[1] if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(im_sets[0], images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(im_sets[2], 0, 7.0, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(im_sets[3], image)))

    cv2.destroyAllWindows()
    video.release()


# ----------



# image_folder = os.path.join(os.getcwd(),'6_Project/data/processed/2018-08-01-11-13-14_left/')
# allfiles = sorted(os.listdir(image_folder))
# video_name = 'video_left.avi'

# images = [img for img in allfiles if img.endswith(".jpg")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 7.0, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()

# # depth maps
# image_folder = os.path.join(os.getcwd(),'6_Project/data/results/')
# images_sorted = sorted(os.listdir(image_folder))
# video_name = 'my_depth_map.avi'

# images = [img for img in images_sorted if img.endswith(".jpg")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 7.0, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()