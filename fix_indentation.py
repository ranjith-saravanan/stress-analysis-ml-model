import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
in_function = False
function_indent = 0

for i, line in enumerate(lines):
    # Check if this is a function definition at the module level
    if re.match(r'^def (show_|load_data|train_models|main)\w*\(', line):
        in_function = True
        function_indent = 0
        fixed_lines.append(line)
    elif in_function and line.strip() and not line[0].isspace():
        # We've left the function (found non-indented code)
        in_function = False
        fixed_lines.append(line)
    elif in_function and line.strip():
        # Inside function - ensure proper indentation
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        # If line has no indentation but should be inside function, add 4 spaces
        if current_indent == 0 and stripped:
            fixed_lines.append('    ' + stripped)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Indentation fixed!")
