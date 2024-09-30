import os
import pandas as pd
import Path
import numpy as np
import echo_video_processor

# each subfolder contains mp4 or avi videos
video_path = './NO_rawVideo'


study_count = 0

views = ["PSAX_V", "PSAX_M", "PLAX", "AP2", "AP3", "AP4", "A2", "A3", "A4"]

# output path for all frame
output_path = './NC_Echo_frames'

disease_name = 'NC'

for resid in Path(video_path).iterdir():
    
    
    videos = list(resid.glob("*.avi")) + list(resid.glob("*.mp4"))
    print('Processing', resid)
            
    for video in videos:

        for view in views:

            if view in str(video).split(os.sep)[-1]:

                view_name = view

                if view_name == "A2":

                    view_name = "AP2"

                elif view_name == "A3":

                    view_name = "AP3"

                elif view_name == "A4":

                    view_name = "AP4"

                break
        
        if not view_name:
            return "View not supported..."

        if view_name != "AP4":
            continue

        print('Processing study:', resid)
        added_count += 1
        echo_video_processor.list_cycler(str(video), resid, disease_name, view_name, output_path) # error occuring here