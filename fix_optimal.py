#!/usr/bin/python
file_name = 'att48_s.txt'
new_file_name = 'att48_s.csv'
new_file = ''
with open(file_name) as opt:
    for line in opt:
        new = line.replace('\n', ', ')
        new_file += new

# remove final city
new_file = new_file[:(len(new_file)) - 5]

with open(new_file_name, 'a') as opt:
    opt.write(new_file)
