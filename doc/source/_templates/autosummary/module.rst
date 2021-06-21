{{ fullname | escape | underline }}


.. automodule:: {{ fullname }}

   {% if classes %}
   :Classes::

   .. autosummary::

      {% for class in classes %}
      {{ class }}
      {%- endfor %}
   {% endif %}

   {% if functions %}
   :Functions::

   .. autosummary::

      {% for function in functions %}
      {{ function }}
      {%- endfor %}
   {% endif %}


   {% block classes %}
   {% if classes %}

   Classes
   =======

   {% for item in classes %}

   {% set header = item | escape %}
   {% set header = ['*', header, '*'] | join %}
   {{ header | underline(line='-') | indent(3) }}

   .. autoclass:: {{ item }}
      :members:
      :special-members: __init__, __iadd__
      :undoc-members:

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}
       :add-heading: used in

   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block functions %}
   {% if functions %}

   Functions
   =========

   {% for item in functions %}

   {# assignments: https://jinja.palletsprojects.com/en/2.11.x/templates/#assignments #}
   {# filter parameters: https://docs.fortinet.com/document/fortisoar/7.0.0/playbooks-guide/767891/jinja-filters-and-functions #}
   {# indentation: https://stackoverflow.com/a/10997352 #}
   {% set header = item | escape %}
   {% set header = ['*', header, '*'] | join %}
   {{ header | underline(line='-') | indent(3) }}

   .. autofunction:: {{ item }}

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}
      :add-heading: used in

   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block exceptions %}
   {% if exceptions %}

   Exceptions
   ==========

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

