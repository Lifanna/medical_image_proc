# -*- coding: utf-8 -*-
from django.conf.urls import url
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import logout
from django.conf.urls.static import static
from django.conf import settings
from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.index, name="index"),
    path('predict_illness', views.predict_illness,),
    path('signin/', views.signin, name='signinview'),
    path('login/', views.login, name='loginview'),
    path('signup/', views.signup, name='signupview'),
    path('signout/', views.logout, name='signoutview'),
    path('load_more_images/', views.load_more_images, name='loadmoreimagesview'),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# if DEBUG:
#     import debug_toolbar
#     urlpatterns = [
#         path('__debug__/', include(debug_toolbar.urls)),
#     ] + urlpatterns

# handler404 = 'main.views.handler404'
# handler500 = 'main.views.handler500'
