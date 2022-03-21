'''
Auxiliary functions for PyWake GUI definition
-
-
-
-

'''
#Import Python modules
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg   #requires 'pip install pysimplegui'
from skimage import measure #requires 'pip install scikit-image'
from mpl_toolkits.mplot3d import axes3d

#Import PyWake modules
import geom_functions as PyW_geo
import plot_functions as PyW_plt

THEME='LightBlue'

def progress_bar():
    sg.theme(THEME)
    BAR_MAX = 1000

    # layout the Window
    layout = [[sg.Text('A custom progress meter')],
              [sg.ProgressBar(BAR_MAX, orientation='h', size=(20,20), key='-PROG-')],
              [sg.Cancel()]]

    # create the Window
    window = sg.Window('Custom Progress Meter', layout)
    # loop that would normally do something useful
    for i in range(1000):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=10)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            break

        # update bar with loop value +1 so that bar eventually reaches the maximum
        window['-PROG-'].update(i+1)
    # done with loop... need to destroy the window as it's still open
    window.close()

def menu():
    sg.theme(THEME)
    sg.set_options(element_padding=(0, 0))      

    # ------ Menu Definition ------ #      
    menu_def = [['File', ['Open', 'Save', 'Exit'  ]],      
                ['Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],      
                ['Help', 'About...'], ]      

    # ------ GUI Defintion ------ #      
    layout = [      
        [sg.Menu(menu_def, )],      
        [sg.Output(size=(60, 20))]      
             ]      

    window = sg.Window("Windows-like program", layout, default_element_size=(12, 1), auto_size_text=False, auto_size_buttons=False,      
                       default_button_element_size=(12, 1))      

    # ------ Loop & Process button menu choices ------ #      
    while True:      
        event, values = window.read()      
        if event == sg.WIN_CLOSED or event == 'Exit':      
            break      
        print('Button = ', event)      
        # ------ Process menu choices ------ #      
        if event == 'About...':      
            sg.popup('About this program', 'Version 1.0', 'PySimpleGUI rocks...')      
        elif event == 'Open':      
            filename = sg.popup_get_file('file to open', no_window=True)      
            print(filename)  

def tabs():
    tab1_layout =  [[sg.T('This is inside tab 1')]]    

    tab2_layout = [[sg.T('This is inside tab 2')],    
                   [sg.In(key='in')]]    

    layout = [[sg.TabGroup([[sg.Tab('Tab 1', tab1_layout, tooltip='tip'), sg.Tab('Tab 2', tab2_layout)]], tooltip='TIP2')],    
              [sg.Button('Read')]]    

    window = sg.Window('My window with tabs', layout, default_element_size=(12,1))    

    while True:    
        event, values = window.read()    
        print(event,values)    
        if event == sg.WIN_CLOSED:           # always,  always give a way out!    
            break

def plot_window():
    '''
    Simultaneous PySimpleGUI Window AND a Matplotlib Interactive Window
    A number of people have requested the ability to run a normal PySimpleGUI window that
    launches a MatplotLib window that is interactive with the usual Matplotlib controls.
    It turns out to be a rather simple thing to do.  The secret is to add parameter block=False to plt.show()
    '''

    layout = [[sg.Button('Plot'), sg.Cancel(), sg.Button('Popup')]]

    window = sg.Window('Have some Matplotlib....', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'Plot':
            draw_plot() #change for the desired function
        elif event == 'Popup':
            sg.popup('Yes, your application is still running')
    window.close()

def columns():
    '''
    Demo of how columns work      
    GUI has on row 1 a vertical slider followed by a COLUMN with 7 rows    
    Prior to the Column element, this layout was not possible      
    Columns layouts look identical to GUI layouts, they are a list of lists of elements.    
    '''
    sg.theme(THEME)      

    # Column layout      
    col = [[sg.Text('col Row 1', text_color='white', background_color='blue')],      
           [sg.Text('col Row 2', text_color='white', background_color='blue'), sg.Input('col input 1')],      
           [sg.Text('col Row 3', text_color='white', background_color='blue'), sg.Input('col input 2')]]      

    layout = [[sg.Listbox(values=('Listbox Item 1', 'Listbox Item 2', 'Listbox Item 3'), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(20,3)), sg.Column(col, background_color='blue')],      
              [sg.Input('Last input')],      
              [sg.OK()]]      

    # Display the Window and get values    

    event, values = sg.Window('Compact 1-line Window with column', layout).Read()  

    sg.popup(event, values, line_width=200) 