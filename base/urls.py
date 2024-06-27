from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_view
from .forms import LoginForm

urlpatterns = [
    path('', views.Home.as_view(template_name='base/inicio.html'), name='inicio'),

    #Auntenticacion
    path('login/', auth_view.LoginView.as_view(template_name='base/login.html', authentication_form=LoginForm), name='login'),
    path('logout/', auth_view.LogoutView.as_view(next_page = 'base:login'), name='logout'),
    path('informedisc/<slug:val>', views.ResultadoDisc.as_view(template_name='base/informedisc.html'), name='informe_disc'),
    path('informediscCare/<slug:val>', views.ViewGraficoCare.as_view(template_name='base/informediscCare.html'), name='grafico_care'),
    path('informediscword/<slug:val>', views.DescargarWord.as_view(), name='informe_dcl'),

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
