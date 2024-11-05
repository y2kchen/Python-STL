
# Iterate over each saved line to extract characters
for line_filename in os.listdir(output_dir):
    line_img = cv2.imread(os.path.join(output_dir, line_filename), cv2.IMREAD_GRAYSCALE)
    
    # Sum pixels horizontally within each line (horizontal projection)
    horizontal_sum = np.sum(line_img, axis=1)
    threshold = np.max(horizontal_sum) * 0.2  # Tune threshold

    # Find start and end of each character within the line
    character_regions = []
    in_character = False
    for j, value in enumerate(horizontal_sum):
        if value > threshold and not in_character:
            start = j
            in_character = True
        elif value <= threshold and in_character:
            end = j
            in_character = False
            character_regions.append((start, end))

    # Save each character as an individual image
    for k, (start, end) in enumerate(character_regions):
        character_img = line_img[start:end, :]
        character_img = cv2.resize(character_img, (64, 64))  # Resize for uniformity
        cv2.imwrite(f'{output_dir}/character_{line_filename}_{k}.png', character_img)
