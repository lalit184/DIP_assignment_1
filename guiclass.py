from tkinter import *
from tkinter.filedialog import askopenfilename,asksaveasfile
import cv2
import numpy as np
from PIL import Image
import PIL.ImageTk

class GUI():
	def __init__(self):
		
		self.master=Tk()
		self.width=900
		self.height=900
		self.dimensions=str(self.height)+"x"+str(self.width)
		self.filename=""
		self.img=None
		self.image_stack=None
		self.stack_count=0
		self.max=0
		self.revert_flag=0

		self.master.geometry(self.dimensions)
		
		self.button_declarations()
		

		self.load.pack(side=BOTTOM,padx=0,pady=0)
		

	def button_declarations(self):
		self.load =  Button(self.master,text="load image",command=self.get_filename)
		self.save = Button(self.master,text="save image",command=self.save_file)
		
		self.undo_button=Button(self.master,text="undo",command=self.undo)
		self.redo_button=Button(self.master,text="redo",command=self.redo)
		self.revert_all=Button(self.master,text="revert all",command=self.revert)
		
		self.histogram_e=Button(self.master,text="equalize histogram",command=self.histogram_equalization)
		self.gamma_c = Button(self.master,text="gamma transform",command=self.gamma_transform)
		self.log_t = Button(self.master,text="log transform",command=self.log_transform)
		self.bluring_image= Button(self.master,text="blurring",command=self.bluring)
		self.sharpening_image= Button(self.master,text="sharpen (0.0<vaulue<1.0) ",command=self.sharpening)
		self.Entry_fields()

	def Entry_fields(self):
		self.gamma_value = Entry(self.master)
		self.sharpen_value = Entry(self.master)
		self.blur_value = Entry(self.master)
			
	def button_placement(self):
		self.save.pack(side=BOTTOM,padx=50,pady=0)
		self.undo_button.place(x=300,y=600)
		self.redo_button.place(x=500,y=600)
		self.revert_all.place(x=700,y=600)
		
		self.histogram_e.place(x=150,y=630)
		self.gamma_c.place(x=150,y=660)
		self.log_t.place(x=150,y=690)
		self.bluring_image.place(x=150,y=720)
		self.sharpening_image.place(x=150,y=750)
		self.gamma_value.place(x=400,y=660)
		self.sharpen_value.place(x=400,y=750)
		self.blur_value.place(x=400,y=720)


	def get_filename(self):

		self.open_filename=askopenfilename()
		self.original=cv2.imread(self.open_filename)
		self.image_stack=np.array([self.original])
		self.current=self.image_stack[self.stack_count,:,:,:]

		self.button_placement()
		
		self.display()
	
	def stack_check(self):
		if self.stack_count<0:
			self.stack_count=0
		(c,a,b,d)=self.image_stack.shape
		if self.stack_count>c-1:
			self.stack_count=c-1


	def display(self):
		self.stack_check()	

		self.current=self.image_stack[self.stack_count,:,:,:]
		self.current_pil = Image.fromarray(cv2.cvtColor(self.current,cv2.COLOR_BGR2RGB))
		self.current_pil = PIL.ImageTk.PhotoImage(self.current_pil)
		
		if self.img:
			self.img.pack_forget()

		self.img = Label(self.master, image=self.current_pil)
		self.img.image = self.current_pil
		self.img.pack(side=TOP,padx=0, pady=0)
		
		self.master.mainloop()
	
	def convolution(self,L,filter):
		filter=np.flip(filter)
		(m,n)=filter.shape  #assuming m,n are odd numbers
		(M,N,c)=self.original.shape
		result=np.zeros((M,N))
				
		image_padded=np.zeros((M+m,N+n))
		image_padded[m//2:M+m//2,n//2:N+n//2]=L

		for i in range(M):
			for j in range(N):
				result[i,j]=np.mean(np.multiply(filter,image_padded[i:i+m,j:j+n]))
		return result		

	def bluring(self):
		
		L,a,b = cv2.split(cv2.cvtColor(self.current,cv2.COLOR_BGR2LAB))
		window_size = self.blur_value.get()
		window_size = int(float(window_size))
		L = self.convolution(L,np.ones((window_size,window_size))).astype("uint8")
		
		self.image_stack = np.insert(self.image_stack,obj=self.stack_count+1,values=cv2.cvtColor(cv2.merge([L,a,b]),cv2.COLOR_LAB2BGR),axis=0)
		self.stack_count+=1
		self.revert_flag=0
		self.display()		

	def sharpening(self):
		L,a,b=cv2.split(cv2.cvtColor(self.current,cv2.COLOR_BGR2LAB))
		
		kernelx=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
		kernely=np.flip(np.transpose(kernelx),axis=0)

		x=self.convolution(L,kernelx)
		x=np.multiply(x,x)
		y=self.convolution(L,kernely)
		y=np.multiply(y,y)
		L=np.clip(L+float(self.sharpen_value.get())*np.clip(2*np.sqrt(x+y).astype("uint8"),a_min=0,a_max=255),a_min=0,a_max=255).astype("uint8")

		self.image_stack = np.insert(self.image_stack,obj=self.stack_count+1,values=cv2.cvtColor(cv2.merge([L,a,b]),cv2.COLOR_LAB2BGR),axis=0)
		self.stack_count+=1
		self.revert_flag=0
		self.display()	
			
		
	def histogram_equalization(self):
		L,a,b=cv2.split(cv2.cvtColor(self.current,cv2.COLOR_BGR2LAB))
		
		integral=np.cumsum(np.histogram(a=L,bins=np.arange(257),density=True)[0])
		L=[255.0*integral[x] for x in L]
		L=np.array(L).astype("uint8")
		
		self.image_stack = np.insert(self.image_stack,obj=self.stack_count+1,values=cv2.cvtColor(cv2.merge([L,a,b]),cv2.COLOR_LAB2BGR),axis=0)
		self.stack_count+=1
		self.revert_flag=0
		self.display()

	def log_transform(self):
		L,a,b=cv2.split(cv2.cvtColor(self.current,cv2.COLOR_BGR2LAB))
		L=255.0*np.log(1+L.astype("float32")/255.0)
		L=np.array(L).astype("uint8")
		
		self.image_stack = np.insert(self.image_stack,obj=self.stack_count+1,values=cv2.cvtColor(cv2.merge([L,a,b]),cv2.COLOR_LAB2BGR),axis=0)
		self.stack_count+=1
		self.revert_flag=0
		self.display()

		
	def gamma_transform(self):
		L,a,b=cv2.split(cv2.cvtColor(self.current,cv2.COLOR_BGR2LAB))
		L=255*((L/255.0)**float(self.gamma_value.get()))
		L=L.astype("uint8")
		self.image_stack = np.insert(self.image_stack,obj=self.stack_count+1,values=cv2.cvtColor(cv2.merge([L,a,b]),cv2.COLOR_LAB2BGR),axis=0)
		self.stack_count+=1
		self.revert_flag=0
		self.display()		

	
		
	def undo(self):
		self.stack_count-=1
		self.display()
	
	def redo(self):
		self.stack_count+=1
		self.display()

	def revert(self):
		if self.revert_flag ==0:
			self.image_stack = np.insert(self.image_stack,obj=self.stack_count+1,values=self.original,axis=0)
			self.stack_count+=1
			self.revert_flag=1
			self.display()
		else:
			self.revert_flag=0
			self.display()	
	def save_file(self):
		cv2.imwrite("result.png",self.current)
		self.master.mainloop()
	

	def execute(self):		
		self.master.mainloop()
gui=GUI()
gui.execute()		