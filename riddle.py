# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def noidea():
    arr1 = np.array(["noooooone","noooooone","noooooone","noooooone"])
    arr2 = np.array(["noooooone","noooooone","noooooone","noooooone"])
    arr3 = np.array(["noooooone","noooooone","noooooone","noooooone"])
    arr4 = np.array(["noooooone","noooooone","noooooone","noooooone"])
    sol = np.array([arr1, arr2, arr3, arr4])


    obj1 = np.array(["eng", "photo", "science","doc"])
    obj2 = np.array(["sun", "rain", "thunder","cloud"])
    obj3 = np.array(["dog", "rabbit", "mouse","lion"])
    obj4 = np.array(["blue", "brown", "gold","school"])
    riddle = np.array([obj1, obj2, obj3, obj4])

    clues = np.array(["rain - blue", "mouse + blue", "thunder + lion", "sun - gold", "thunder - brown", "dog - blue", "lion + gold",
                      "ruler - thunder", "sun + school", "photo - school", "lion - blue", "science - blue", "rabbit - brown", "doctor + crown", "science - school"])


    sol[3] = riddle[3]

    print(sol[3])
    print(riddle[3])
    for x in clues:
        txt = x.split()
        if txt[1] == "-":
            y=1
        elif txt[1]== "+":
            y=1






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    noidea()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
