from django.shortcuts import render
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from .detector import detect
from django.contrib.auth import authenticate, login as authlogin, logout as authlogout
from .forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import Image
from django.http.response import JsonResponse
from datetime import datetime
import os


@login_required(login_url='/signin/')
@csrf_exempt
def index(request):
    images_count = len(Image.objects.all())

    return render(request, 'img_proc_app/index.html', context={'images_count': images_count})

@login_required(login_url='/signin/')
@csrf_exempt
def load_more_images(request):
    images_count = len(Image.objects.all())
    images = Image.objects.all()[int(request.GET.get('offset')):int(request.GET.get('offset')) + 5].values()

    return JsonResponse({
        'images': list(images),
        'images_count': images_count
    })

@login_required(login_url='/signin/')
@csrf_exempt
def predict_illness(request):
    fileName, fileExtension = os.path.splitext(request.POST.get('filename'))

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    image = Image(path = "images/%s/"%date_time, image_file=request.FILES['file'])
    image.save()

    uploaded_path = image.image_file.url

    try:
        result = detect(fileName)

        return JsonResponse({
            'imageID': str(image.id),
            'path': str(image.path),
            'checked': str(image.checked),
            'result': str(result),
            'created_at': str(image.created_at),
            'updated_at': str(image.updated_at),
            'image_file': str(image.image_file),
        })
    except Exception as e:
        import traceback, sys
        print(traceback.format_exc())
        if type(e).__name__ == "InvalidArgumentError":
            return JsonResponse({'error': 'err', 'message': 'Please, check your image and try again'})

        # print(sys.exc_info()[2])

        return JsonResponse({'error': 'err', 'message': 'Internal server error'})

def signin(request):
    return render(request, 'img_proc_app/login.html')

def login(request):
    username = request.POST['login']
    password = request.POST['password']
    user = authenticate(request, username=username, password=password)
    
    if user is not None:
        authlogin(request, user)

        return redirect('index')

    return render(request, 'img_proc_app/login.html', {'error': 'Incorrect username or password'})

def logout(request):
    authlogout(request)

    return redirect('index')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(request, username=username, password=raw_password)
            authlogin(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()


    return render(request, 'img_proc_app/signup.html', {'form': form})
