
# Specify the input and output file paths
input_file = 'letter_sample_2D-1b_transcript.txt'  # Replace with your file path
output_file = 'output.txt'  # The file where reversed content will be saved

# Read the file and reverse the lines
with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()  # Read all lines into a list

# Reverse the order of lines
reversed_lines = lines[::-1]

# Write the reversed lines to a new file
with open(output_file, 'w', encoding='utf-8') as file:
    file.writelines(reversed_lines)

print(f"The reversed content has been saved to {output_file}.")
