from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

root = Tk()
root.title("Digital Image Processor Project")
root.geometry("500x500")

global my_label2,my_label1
my_label2 = Label(root)
my_label1 = Label(root)
my_label2.grid(row =1,column = 50)
my_label1.grid(row =1,column = 0)
# my_label3 = Label(root)
# my_label3.grid(row=1,column=50)

global kernel_MT
kernel_MT = np.ones((5,5),np.uint8)

def output_image(img):
    global my_label2
    my_label2.destroy() #Make function
    my_label2 = Label(root,image = img)
    my_label2.grid(row = 1, column = 50)
    
def clear_command():
    # my_label2.grid_forget()
    # my_label1.grid_forget()
    
    my_label2.destroy()
    my_label1.destroy()
    # root.filename = NONE
    # my_label3.destroy()
    # my_label1.after(1000,my_label1.destroy())
    # my_label2.after(1000,my_label2.destroy())
   
def file_open():
    global my_img,my_label1,my_label2
    global img
    root.filename = filedialog.askopenfilename(initialdir='Photos',title="Select a file",filetypes=(("jpg files","*.jpg"),("All files","*.*")))
    my_img = ImageTk.PhotoImage(Image.open(root.filename))
    clear_command()
    my_label1.destroy()
    my_label1 = Label(root,image = my_img)
    my_label1.grid(row = 1,column = 0,columnspan=3)
    img = cv.imread(root.filename)
    

    
    
def cvt_Gray():
    # global recolour
    global image2,my_label2,img
    img = cv.imread(root.filename)
    recolour = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    image2 = ImageTk.PhotoImage(Image.fromarray(recolour))
    output_image(image2)
    
def cvt_RGB():
    # global recolour
    global image,my_label2
    img = cv.imread(root.filename)
    recolour = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    recolour = cv.cvtColor(recolour, cv.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(Image.fromarray(recolour))
    # my_label2.grid_forget()
    output_image(image)
    
def cvt_Histogram():
    global img,my_label2
    img = cv.imread(root.filename)
    plt.figure()
    plt.title('Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colour = ('b','g','r')
    for i,col in enumerate(colour):
        hist = cv.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    
    
    Htg_1 = plt.show()
    
    
def Mean_filter():
    global m_img,my_label2
    m_img = cv.blur(img,(5,5))
    m_img = cv.cvtColor(m_img, cv.COLOR_BGR2RGB)
    m_img = ImageTk.PhotoImage(Image.fromarray(m_img))
    output_image(m_img)

def Median_filter():
    global median_img,my_label2
    median_img = cv.medianBlur(img,7)
    median_img = cv.cvtColor(median_img, cv.COLOR_BGR2RGB)
    median_img = ImageTk.PhotoImage(Image.fromarray(median_img))
    output_image(median_img)

def G_filter():
    global my_label2,blur_img
    blur_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(img, ksize = (5,5), sigmaX = 30,sigmaY = 300)
    blur_img = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)
    blur_img = ImageTk.PhotoImage(Image.fromarray(blur_img))
    output_image(blur_img)

def Sobel_filter():
    global combined_sobel,my_label2
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray,cv.CV_64F, 1, 0)
    sobely = cv.Sobel(gray,cv.CV_64F, 0, 1)
    combined_sobel = cv.bitwise_or(sobelx,sobely)
    combined_sobel = ImageTk.PhotoImage(Image.fromarray(combined_sobel))
    output_image(combined_sobel)

def Prewitt_filter():
    global combined_prewitt,my_label2
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    g_img = cv.GaussianBlur(gray, (3,3), 0)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv.filter2D(g_img, -1, kernelx)
    img_prewitty = cv.filter2D(g_img, -1, kernely)
    combined_prewitt = img_prewittx + img_prewitty
    combined_prewitt = ImageTk.PhotoImage(Image.fromarray(combined_prewitt))
    output_image(combined_prewitt)

def Canny_filter():
    global my_label2,resized_img2
    img = cv.imread(root.filename)
    blur = cv.GaussianBlur(img, (3,3), 0)
    canny_img = cv.Canny(blur,threshold1=205,threshold2=210)
    resized_img = cv.resize(canny_img, (int(img.shape[1]), int(img.shape[0])))
    # resized_img2 = ImageTk.PhotoImage(Image.fromarray(resized_img))
    resized_img2 = ImageTk.PhotoImage(Image.fromarray(resized_img))
    output_image(resized_img2)
    
def Fourier_T():
    global my_label2,g_img,P_img,magnitude
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    f = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(cv.magnitude(fshift[:,:,0],fshift[:,:,1]))
    # Scale the magnitude for display
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    magnitude = ImageTk.PhotoImage(Image.fromarray(magnitude))
    output_image(magnitude)
    
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()


# def M_T():
#     pass

def Erosion():
    global my_label2,erode_img
    erode_img = cv.erode(img,kernel = kernel_MT,iterations = 1)
    erode_img = cv.cvtColor(erode_img,cv.COLOR_BGR2RGB)
    erode_img = ImageTk.PhotoImage(Image.fromarray(erode_img))
    output_image(erode_img)

def Dilation():
    global my_label2,d_img
    d_img = cv.dilate(img,kernel = kernel_MT,iterations = 1)
    d_img = cv.cvtColor(d_img,cv.COLOR_BGR2RGB)
    d_img = ImageTk.PhotoImage(Image.fromarray(d_img))
    output_image(d_img)
def Opening():
    global my_label2,o_img
    o_img = cv.morphologyEx(img,cv.MORPH_OPEN,kernel_MT)
    o_img = cv.cvtColor(o_img,cv.COLOR_BGR2RGB)
    o_img = ImageTk.PhotoImage(Image.fromarray(o_img))
    output_image(o_img)

def Closing():
    global my_label2,c_img
    c_img = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel_MT)
    c_img = cv.cvtColor(c_img,cv.COLOR_BGR2RGB)
    c_img = ImageTk.PhotoImage(Image.fromarray(c_img))
    output_image(c_img)

def Hit_Miss():
    pass

def Segmentation():
    global my_label2,segmented_image
    s_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    pixel_vals = s_img.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 10
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((s_img.shape))
    segmented_image = ImageTk.PhotoImage(Image.fromarray(segmented_image))
    output_image(segmented_image)

def Contour_Detection():
    global image_copy,my_label2
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    image_copy = img.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    image_copy = ImageTk.PhotoImage(Image.fromarray(image_copy))
    output_image(image_copy)

my_menu = Menu(root)
root.config(menu = my_menu)

#File Menu
file_menu  = Menu(my_menu)
my_menu.add_cascade(label = "File",menu = file_menu)
file_menu.add_command(label = "New...",command = file_open)
file_menu.add_command(label = "Clear",command = clear_command)
file_menu.add_separator()
file_menu.add_command(label = "Exit",command = root.quit)

#Convert Menu
convert_menu = Menu(my_menu)
my_menu.add_cascade(label = "Convert",menu = convert_menu)
convert_menu.add_command(label = "Gray",command = cvt_Gray)
convert_menu.add_command(label = "RGB",command = cvt_RGB)
convert_menu.add_command(label = "Histogram",command = cvt_Histogram)

#Filter Menu
filter_menu = Menu(my_menu)
my_menu.add_cascade(label = "Filters",menu = filter_menu)
filter_menu.add_command(label = "Mean Filter",command = Mean_filter)
filter_menu.add_command(label = "Median Filter",command = Median_filter)
filter_menu.add_command(label = "Gaussian Filter",command = G_filter)
filter_menu.add_separator()
filter_menu.add_command(label = "Sobel Filter",command = Sobel_filter)
filter_menu.add_command(label = "Prewitt Filter",command = Prewitt_filter)
filter_menu.add_command(label = "Canny Filter",command = Canny_filter)

#Transformation Menu
Transformation_menu = Menu(my_menu)
my_menu.add_cascade(label = "Transformation",menu = Transformation_menu)
Transformation_menu.add_command(label = "Fourier Transformation",command = Fourier_T)
# Transformation_menu.add_command(label = "Morphological Transformation",command = M_T)
# Transformation_menu.add_cascade(label = "Erosion", menu = Transformation_menu)

#Morphological Transformation Menu
M_Transformation_menu = Menu(my_menu)
my_menu.add_cascade(label = "Morphological Transformation",menu = M_Transformation_menu)
M_Transformation_menu.add_command(label = "Erosion",command = Erosion)
M_Transformation_menu.add_command(label = "Dilation",command = Dilation)
M_Transformation_menu.add_command(label = "Opening",command = Opening)
M_Transformation_menu.add_command(label = "Closing",command = Closing)

#Segmentation Menu
Segmentation_menu = Menu(my_menu)
my_menu.add_cascade(label = "Segmentation",menu = Segmentation_menu)
Segmentation_menu.add_command(label = "Using K-Means",command = Segmentation)
Segmentation_menu.add_command(label = "Contour Detection",command = Contour_Detection)
root.mainloop()