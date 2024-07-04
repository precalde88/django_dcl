from typing import Any
from django.http import HttpRequest
from django.http.response import HttpResponse as HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.contrib.auth.mixins import LoginRequiredMixin,\
     PermissionRequiredMixin
from django.views import generic
from . import GetDataframe
import imgkit
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
from django.conf import settings
import os
import tempfile
import datetime
import zoneinfo

zona_asuncion = zoneinfo.ZoneInfo("America/Asuncion")

# Create your views here.
class Home(LoginRequiredMixin, generic.TemplateView):
    login_url='base:login'
    def get(self, request):
        total = GetDataframe.df_info_inicial()
        columns = total.columns.values
        dict_total = total.to_dict('records')
        return render(request, 'base/disctotal.html', locals())

    
class ResultadoDisc(LoginRequiredMixin, generic.TemplateView):
    def get(self, request, val):
        val_int = int(val)
        val_int -= 1
        total = GetDataframe.carga_total_completo(val_int)
        total_ = total.iloc[:,10:14].T
        disc_list_pc = total_[0].values.tolist()
        care_list = GetDataframe.list_care_for_graf(val_int)
        total.columns = total.columns.str.replace(" ", "_")
        dict_total = total.to_dict('records')
        #valor_url = request.build_absolute_uri(reverse('base:grafico_care', args=[val_int]))
        graf_disc = GetDataframe.get_disc_graf(val_int)
        graf_care = GetDataframe.get_grafico_polar_care_render(val_int)
        graf_lider = GetDataframe.get_grafico_polar_liderazgo_render(val_int)
        """ con = imgkit.config(wkhtmltoimage='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltoimage.exe')
        imgkit.from_url(valor_url , 'out.png', config=con) """
        return render(request, 'base/informedisc.html', locals())


class ViewGraficoCare(generic.TemplateView):
    def get(self, request, val):
        val_int = int(val)
        total = GetDataframe.carga_total_completo(val_int)
        total.columns = total.columns.str.replace(" ", "_")
        care_list = GetDataframe.list_care_for_graf(val_int)
        dict_total = total.to_dict('records')
        return render(request, 'base/informediscCare.html', locals())
    

class DescargarWord(generic.TemplateView):
    def get(self, request, val):
        val_int = int(val)
        hora_asuncion = datetime.datetime.now(zona_asuncion)
        hora_asuncion = hora_asuncion.strftime('%d/%m/%Y')
        info_test = GetDataframe.info_test_total(val_int)
        name = info_test['Nombre y apellido'].values.tolist()
        #Dict info total DCL
        total = GetDataframe.carga_total_completo(val_int)
        total.columns = total.columns.str.replace(" ", "_")
        dict_total = total.to_dict('records')
        dict_total_fin = dict_total[0]
        response = HttpResponse(content_type='application/msword')
        response['Content-Disposition'] = f'attachment; filename="{name[0]}.docx"'
        
        path_plantilla = os.path.join(settings.BASE_DIR,'base', 'plantilla', 'Plantilla_Informe.docx') 
        doc = DocxTemplate(path_plantilla)
        graf_care = GetDataframe.get_grafico_polar_care_word(val_int)
        graf_lider = GetDataframe.get_grafico_polar_liderazgo_word(val_int)
        graf_disc = GetDataframe.get_disc_word(val_int)
        fp = tempfile.NamedTemporaryFile()
        with open(f"{fp.name}.png", 'wb') as temp_file:
            temp_file.write(graf_care)
            img_temp = str(temp_file.name)
            imagen = InlineImage(doc, img_temp, width=Mm(90), height=Mm(85))
        fp2 = tempfile.NamedTemporaryFile()
        with open(f"{fp2.name}.png", 'wb') as temp_file2:
            temp_file2.write(graf_lider)
            img_temp = str(temp_file2.name)
            imagen2 = InlineImage(doc, img_temp, width=Mm(90) )
        fp3 = tempfile.NamedTemporaryFile()
        with open(f"{fp3.name}.png", 'wb') as temp_file3:
            temp_file3.write(graf_disc)
            img_temp = str(temp_file3.name)
            imagen3 = InlineImage(doc, img_temp, width=Mm(60))
        nombre = f"{request.user.first_name} {request.user.last_name}"
        context = {'imagen': imagen, 'dict_total_fin': dict_total_fin, 'imagen2': imagen2, 'imagen3': imagen3, 'hora': hora_asuncion, 'user': nombre}
        doc.render(context)
        doc.save(response)
        return response
