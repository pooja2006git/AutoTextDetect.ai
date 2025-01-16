from google.cloud import vision
import io
import os

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'VisionToken.json'

def detect_text_with_alignment(image_path):
    # Initialize the Vision API client
    client = vision.ImageAnnotatorClient()

    # Read the image for OCR
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    # Extract text annotations
    annotations = response.text_annotations
    if not annotations:
        print("No text detected.")
        return

    # Extract words and their bounding boxes
    words = []
    for annotation in annotations[1:]:  # Skip the first annotation (full text)
        vertices = annotation.bounding_poly.vertices
        words.append({
            "text": annotation.description,
            "bounding_box": {
                "x_min": min(vertex.x for vertex in vertices),
                "x_max": max(vertex.x for vertex in vertices),
                "y_min": min(vertex.y for vertex in vertices),
                "y_max": max(vertex.y for vertex in vertices),
            }
        })

    # Group words into lines based on their y-coordinates
    lines = []
    for word in words:
        added_to_line = False
        for line in lines:
            # Check if the word belongs to the current line
            if abs(word["bounding_box"]["y_min"] - line["y_min"]) < 15:  # Threshold for line grouping
                line["words"].append(word)
                line["y_min"] = min(line["y_min"], word["bounding_box"]["y_min"])
                line["y_max"] = max(line["y_max"], word["bounding_box"]["y_max"])
                added_to_line = True
                break
        if not added_to_line:
            # Create a new line if the word doesn't fit into existing lines
            lines.append({
                "words": [word],
                "y_min": word["bounding_box"]["y_min"],
                "y_max": word["bounding_box"]["y_max"],
            })

    # Sort lines by their vertical position (y_min)
    lines.sort(key=lambda line: line["y_min"])

    # Construct the text line by line
    print("Reconstructed Text with Proper Alignment:")
    for line in lines:
        # Sort words in the line by their horizontal position (x_min)
        sorted_words = sorted(line["words"], key=lambda word: word["bounding_box"]["x_min"])
        line_text = " ".join(word["text"] for word in sorted_words)
        print(line_text)

    # Handle errors
    if response.error.message:
        raise Exception(f"API Error: {response.error.message}")

# Input image path
image_path = "h1.jpeg"

# Perform OCR with improved alignment
detect_text_with_alignment(image_path)
