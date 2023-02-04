import re


def extract_dates(line, regex):
    
    # Test the regular expression
    match = re.search(regex, line)
    if match is not None:
        date, hour =match.group(1), match.group(2)
        new_line = line.split(match.group(2) + ' - ')[1]
    else:
        date, hour = '', ''
        new_line = ": " + line
    return date, hour, new_line

def extract_name(line):
    
    line_split = line.split(':',1)
    if len(line_split) > 1:
        name = line_split[0]
        new_line = line_split[1]
    else:
        name = ''
        new_line = line
    return name, new_line

def exteact_message(line):
    
    new_line = line.replace("\n", "")
    return new_line
