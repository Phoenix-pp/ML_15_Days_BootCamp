# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:03:42 2023

@author: Pooja
"""
s="this sj,kfjlABCkjsdfhii ABCghoeri"
s1=s.replace("ABC", "GHTTS")

s1.__contains__("ABC")

i1=s.find("ABC")
n=len(s)
s.find("ABC", i1+1,n)

def count_substring(string, sub_string):
    count=0
    for i in range(len(string)-len(sub_string)+1):
        if (string[i:i+len(sub_string)] == sub_string):
            count=count+1
    return count

string1 = input().strip()
sub_string = input().strip()
count = count_substring(string1, sub_string)
print(count)



def solve(s):
    l=list(s)
    l1=[]
    for i in range(len(l)):
        lw=l[i].split()
        lw[0]= lw[0].capitalize()
        lw1="".join(lw)
        l1.append(lw1)
    return " ".join(l1)


s="sajgfu asjdfh skjdh sjfkfdfej"
l=s.split(" ")
l1=[]
for i in range(len(l)):
    lw=l[i].split()
    lw[0]= lw[0].capitalize()
    lw1="".join(lw)
    l1.append(lw1)




















def swap_case(s):
    so=[]
    for i in range(len(s)):
        #i=str(s[i])
        if s[i].isupper():
            so.append(s[i].lower())
        elif s[i].islower():
            so.append(s[i].upper())
        elif s[i].isdigit():
            so.append(s[i])
        else:
            so.append(s[i])
            
    return "".join(so)

s = input()
result = swap_case(s)
print(result)
"""
Sample Input

this is a string   

Sample Output

this-is-a-string
"""
def split_and_join(line):
    # write your code here
    l=line.split(" ")
    return "-".join(l)

line = input()
result = split_and_join(line)
print(result)

#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    # Write your code here
    s="Hello {} {}! You just delved into python.".format(first, last)
    print(s)

first_name = input()
last_name = input()
print_full_name(first_name, last_name)
    
    
def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    s="".join(l)
    return s

s = input()
i, c = input().split()
s_new = mutate_string(s, int(i), c)
print(s_new)