from django.views.decorators.http import require_POST
from django.shortcuts import render, redirect
# from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import UserRegisterForm, xray_imageForm
from .models import xray_image
import os, datetime


# Create your views here.

def index(request):
    # print("hello")
    return render(request, "pneumonia/index.html")

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request,f'Your account has been created ! You are now able to log in.')
            return redirect('pneumonia-login')
    else:
        form = UserRegisterForm()
    return render(request, "pneumonia/register.html", {'form': form}) 



def upload_image_view(request):

    if request.method == "POST":
        form = xray_imageForm(request.POST, request.FILES)
        
        if form.is_valid():
            print('hlw')
            # instance = xray_image(photo = request.FILES['myfile'])
            # instance.save()
            form.save()
            return redirect('pneumonia-home')
        else:
            print(form.errors)
    else:
        form =xray_imageForm()
    return render(request, 'pneumonia/image_form.html', {'form':form})

def history(request):
    history = xray_image.objects.filter(uploaded_by=request.user.username)
    return render(request, "pneumonia/history.html", {'history':history})



    # image = request.FILES.get('myfile')
    # print(image)
    # image_list = xray_image.objects.Create(image=xray_image)
    # image_list.save()
    #     # print(image_list)
    #     # print(image)

    # return render(request, 'pneumonia/image_form.html')




    # if request.method=="POST":
    #     # image = request.POST.get('myfile')
    #     # list=xray_image.objects.Create(image=xray_image)
    #     # list.save()
    #     print("hello")
    #     return redirect(request,"pneumonia/register.html")

def image(request):
    return render(request, 'pneumonia/image_form.html')

def action(request):
    
    file = request.FILES.get("myfile")
    print(file)
    # extensions for files that we want rename
    extensions = (['.jpg', '.jpeg', 'png']);

    # split the file into filename and extension
    filename, extension = os.path.splitext(file)

    # if the extension is a valid extension
    if ( extension in extensions ):
        # Get the create time of the file
        create_time = os.path.getctime( file )
        # get the readable timestamp format 
        format_time = datetime.datetime.fromtimestamp( create_time )
        # convert time into string
        format_time_string = format_time.strftime("%Y-%m-%d %H.%M.%S") # e.g. 2015-01-01 09.00.00.jpg
        # Contruct the new name of the file
        newfile = format_time_string + extension; 

        # rename the file
        os.rename( file, newfile );
        print(newfile)
    