import fiftyone as fo

# fo.utils.video.reencode_video("/vids/IMG_1855.MP4", "/vids/NEW_IMG_1855.MP4")

# Create a dataset from a list of videos
dataset = fo.Dataset.from_videos(
    ["./vids/IMG_1855.MP4"]
)
    
session = fo.launch_app(dataset)

session.wait()