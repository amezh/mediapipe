import json
import os

def read_landmarks_file():
    # Define the file path
    file_path = os.path.join("json", "landmarks_output.json")
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    
    try:
        # Read the JSON file
        with open(file_path, 'r') as file:
            landmarks_data = json.load(file)
            
        print(f"Successfully loaded landmarks data from '{file_path}'")
        
        # Print summary of data
        print(f"Number of frames/objects: {len(landmarks_data)}")
        
        pose_landmarks = []
        def print_first_frame_summary(landmarks_data):
            """
            Print a summary of the landmarks in the first frame.
            """
            if landmarks_data and len(landmarks_data) > 0:
                first_frame = landmarks_data[0]
                print("\nLandmarks in first frame:")
                
                # Check each type of landmark and print count if present
                if "pose_landmarks" in first_frame:
                    print(f"  Pose landmarks: {len(first_frame['pose_landmarks'])}")
                
                if "left_hand_landmarks" in first_frame:
                    print(f"  Left hand landmarks: {len(first_frame['left_hand_landmarks'])}")
                
                if "right_hand_landmarks" in first_frame:
                    print(f"  Right hand landmarks: {len(first_frame['right_hand_landmarks'])}")
                    
                if "face_landmarks" in first_frame:
                    print(f"  Face landmarks: {len(first_frame['face_landmarks'])}")

        # Call the function here
        print_first_frame_summary(landmarks_data)

        
                
        return landmarks_data
        
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    landmarks_data = read_landmarks_file()
    
    if landmarks_data is not None:
        # Further processing can be done here
        print("Landmarks data is ready for further processing.")
    else:
        print("Failed to load landmarks data.")

def process_landmarks_data(landmarks_data):
    # We go frame by frame and process the landmarks

    # We reconstruct the sequence of each landmark for each frame and if "presence" is avaliable check it to be above 0.5
    # for unavaliable landmarks we try to interpolate the gaps
    if not landmarks_data:
        print("No landmarks data to process.")
        return
    for frame_index, frame in enumerate(landmarks_data):
        print(f"Processing frame {frame_index + 1}/{len(landmarks_data)}")
        
        # Example processing logic (to be replaced with actual logic)
        if "pose_landmarks" in frame:
            pose_landmarks = frame["pose_landmarks"]
            # Process pose landmarks here
            
        if "left_hand_landmarks" in frame:
            left_hand_landmarks = frame["left_hand_landmarks"]
            # Process left hand landmarks here
            
        if "right_hand_landmarks" in frame:
            right_hand_landmarks = frame["right_hand_landmarks"]
            # Process right hand landmarks here
            
        if "face_landmarks" in frame:
            face_landmarks = frame["face_landmarks"]
            # Process face landmarks here
    
    

main()