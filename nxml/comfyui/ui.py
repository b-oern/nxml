"""
Workflows im UI-Format von ComfyUI (gespeichert unter user/default/workflows) in das API-Format umwandeln.

Das UI-Format speichert Widget-Werte nur als Liste (widgets_values), die Namen ergeben sich erst aus der
Node-Definition des Servers (/object_info). Unterstuetzt werden Subgraphen (auch verschachtelt),
Bypass und Mute, Reroute, Get/Set-Nodes und die alten PrimitiveNodes.

  conv = UiWorkflow(workflow, object_info)
  conv.params()                    # wichtige Parameter zum Einstellen
  conv.set_param('18', 'seed', 5)  # Wert setzen
  api = conv.to_api()              # {Node-Id: {class_type, inputs, _meta}}
"""
import copy
import random

WIDGET_TYPES = ('INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO')
DYNAMIC_COMBO = 'COMFY_DYNAMICCOMBO_V3'
SUBGRAPH_INPUT = -10
SUBGRAPH_OUTPUT = -20
MODE_MUTED = 2
MODE_BYPASS = 4
SEED_MAX = 2 ** 50

# Nodes auf oberster Ebene, deren (nicht verbundene) Widgets als wichtige Parameter angeboten werden.
# None: alle Widgets
IMPORTANT_NODES = {
    'ResolutionSelector': None,
    'KSampler': ('seed', 'steps', 'cfg'),
    'KSamplerAdvanced': ('noise_seed', 'steps', 'cfg'),
    'CLIPTextEncode': ('text',),
    'PrimitiveString': ('value',),
    'PrimitiveStringMultiline': ('value',),
    'PrimitiveInt': ('value',),
    'PrimitiveFloat': ('value',),
    'PrimitiveBoolean': ('value',),
    'PrimitiveNode': None,
    'LoadImage': ('image',),
    'LoraLoader': ('lora_name', 'strength_model'),
    'LoraLoaderModelOnly': ('lora_name', 'strength_model'),
}
LATENT_PARAMS = ('width', 'height', 'batch_size', 'length')


def spec_options(spec) -> dict:
    return spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}


def spec_type(spec):
    """ :return: 'COMBO' fuer Auswahllisten, sonst der Typ aus object_info """
    t = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else None
    return 'COMBO' if isinstance(t, list) else t


def spec_choices(spec) -> list:
    t = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else None
    if isinstance(t, list):
        return t
    return list(spec_options(spec).get('options', []) or [])


def is_widget(spec) -> bool:
    if spec_options(spec).get('forceInput'):
        return False
    return spec_type(spec) in WIDGET_TYPES or spec_type(spec) == DYNAMIC_COMBO


def default_value(spec):
    """ Standardwert eines Widgets, bei Auswahllisten der erste Eintrag """
    options = spec_options(spec)
    if 'default' in options:
        return options['default']
    choices = spec_choices(spec) if spec_type(spec) == 'COMBO' else []
    return choices[0] if choices else None


def to_type(value, datatype):
    """ Wert aus einem Formular in den Typ des Widgets umwandeln """
    if datatype == 'INT':
        return int(float(value))
    if datatype == 'FLOAT':
        return float(value)
    if datatype == 'BOOLEAN':
        return value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'on', 'yes', 'ja')
    return value if value is None else str(value)


class WidgetLayout:
    """ Position der Widget-Werte in widgets_values fuer eine Node-Klasse (und deren aktuelle Werte) """

    def __init__(self, class_info: dict, values: list):
        self.index = {}     # Name -> Position in widgets_values
        self.specs = {}     # Name -> Spezifikation aus object_info
        self.controls = {}  # Name -> Position von control_after_generate (randomize, fixed, ...)
        self.size = 0
        self._walk(class_info.get('input', {}), class_info.get('input_order'), '', values)

    def _walk(self, inputs: dict, order, prefix, values):
        for section in ('required', 'optional'):
            specs = inputs.get(section) or {}
            names = (order or {}).get(section) or list(specs.keys())
            for name in names:
                spec = specs.get(name)
                if spec is None or not is_widget(spec):
                    continue
                full = prefix + name
                self.index[full] = self.size
                self.specs[full] = spec
                self.size += 1
                options = spec_options(spec)
                # aeltere Nodes ohne Angabe: das Frontend haengt bei seed/noise_seed trotzdem ein Control an
                legacy_seed = 'control_after_generate' not in options and prefix == '' and \
                    spec_type(spec) == 'INT' and name in ('seed', 'noise_seed')
                if options.get('control_after_generate') or legacy_seed:
                    self.controls[full] = self.size
                    self.size += 1
                if spec_type(spec) == DYNAMIC_COMBO:
                    pos = self.index[full]
                    selected = values[pos] if pos < len(values) else None
                    for option in spec_options(spec).get('options', []):
                        if isinstance(option, dict) and option.get('key') == selected:
                            self._walk(option.get('inputs', {}), None, full + '.', values)


class Graph:
    """ Ebene eines Workflows: oberste Ebene oder Inhalt einer Subgraph-Instanz """

    def __init__(self, data: dict, prefix='', parent=None, instance=None):
        self.data = data
        self.prefix = prefix
        self.parent = parent
        self.instance = instance
        self.nodes = {n['id']: n for n in data.get('nodes', [])}
        self.links = {}
        for link in data.get('links', []) or []:
            if isinstance(link, dict):
                self.links[link['id']] = (link['origin_id'], link['origin_slot'], link['target_id'], link['target_slot'])
            elif isinstance(link, list) and len(link) >= 5:
                self.links[link[0]] = (link[1], link[2], link[3], link[4])


class UiWorkflow:

    def __init__(self, workflow: dict, object_info: dict):
        self.workflow = copy.deepcopy(workflow)
        self.object_info = object_info
        self.subgraphs = {s['id']: s for s in self.workflow.get('definitions', {}).get('subgraphs', [])}
        self.warnings = []

    @staticmethod
    def is_ui_format(workflow) -> bool:
        return isinstance(workflow, dict) and isinstance(workflow.get('nodes'), list)

    # --- Widgets ---

    def layout(self, node: dict) -> WidgetLayout:
        values = node.get('widgets_values')
        return WidgetLayout(self.object_info.get(node['type'], {}), values if isinstance(values, list) else [])

    @staticmethod
    def instance_widget_inputs(node: dict) -> list:
        """ Eingaenge einer Subgraph-Instanz, die ein Widget haben (gleiche Reihenfolge wie widgets_values) """
        return [i for i in node.get('inputs', []) if i.get('widget')]

    def widget_values(self, node: dict) -> dict:
        """ Name -> Wert der Widgets eines Nodes """
        values = node.get('widgets_values')
        if isinstance(values, dict):  # manche Custom-Nodes (z.B. VHS) speichern ein Objekt
            return dict(values)
        values = values or []
        if node['type'] in self.subgraphs:
            return {i['name']: values[k] for k, i in enumerate(self.instance_widget_inputs(node)) if k < len(values)}
        if node['type'] == 'PrimitiveNode':
            return {'value': values[0]} if values else {}
        layout = self.layout(node)
        return {name: values[pos] for name, pos in layout.index.items() if pos < len(values)}

    def widget_index(self, node: dict, name: str):
        if node['type'] in self.subgraphs:
            names = [i['name'] for i in self.instance_widget_inputs(node)]
            return names.index(name) if name in names else None
        if node['type'] == 'PrimitiveNode':
            return 0 if name == 'value' else None
        return self.layout(node).index.get(name)

    # --- Verbindungen aufloesen ---

    def resolve_link(self, graph: Graph, link_id, seen=None):
        """ :return: ('node', api_id, slot), ('value', wert) oder None """
        link = graph.links.get(link_id)
        if link is None:
            return None
        return self.resolve_output(graph, link[0], link[1], seen)

    def resolve_output(self, graph: Graph, node_id, slot, seen=None):
        seen = set() if seen is None else seen
        key = (graph.prefix, node_id, slot)
        if key in seen:
            self.warnings.append(f"Zyklus bei Node {graph.prefix}{node_id}")
            return None
        seen.add(key)
        if node_id == SUBGRAPH_INPUT:
            return self.resolve_subgraph_input(graph, slot, seen)
        node = graph.nodes.get(node_id)
        if node is None:
            return None
        mode = node.get('mode', 0)
        if mode == MODE_MUTED:
            return None
        if mode == MODE_BYPASS:
            return self.resolve_bypass(graph, node, slot, seen)
        if node['type'] in self.subgraphs:
            inner = self.inner_graph(graph, node)
            for origin_id, origin_slot, target_id, target_slot in inner.links.values():
                if target_id == SUBGRAPH_OUTPUT and target_slot == slot:
                    return self.resolve_output(inner, origin_id, origin_slot, seen)
            return None
        if node['type'] == 'Reroute':
            inputs = node.get('inputs', [])
            return self.resolve_link(graph, inputs[0].get('link'), seen) if inputs else None
        if node['type'] == 'GetNode':
            return self.resolve_get_node(graph, node, seen)
        if node['type'] == 'PrimitiveNode':
            values = node.get('widgets_values') or []
            return ('value', values[0]) if values else None
        if node['type'] in self.object_info:
            return 'node', graph.prefix + str(node_id), slot
        return None

    def resolve_subgraph_input(self, graph: Graph, slot, seen):
        instance = graph.instance
        if instance is None:
            return None
        sub_inputs = self.subgraphs[instance['type']].get('inputs', [])
        name = sub_inputs[slot]['name'] if slot < len(sub_inputs) else None
        inputs = instance.get('inputs', [])
        inp = next((i for i in inputs if i.get('name') == name), inputs[slot] if slot < len(inputs) else None)
        if inp is None:
            return None
        if inp.get('link') is not None:
            return self.resolve_link(graph.parent, inp['link'], seen)
        if inp.get('widget'):
            values = self.widget_values(instance)
            if inp['name'] in values:
                return 'value', values[inp['name']]
        return None

    def resolve_bypass(self, graph: Graph, node, slot, seen):
        """ Ein uebersprungener Node reicht den ersten Eingang mit dem Typ des Ausgangs durch """
        outputs = node.get('outputs', [])
        out_type = outputs[slot].get('type') if slot < len(outputs) else None
        for inp in node.get('inputs', []):
            if inp.get('link') is not None and inp.get('type') in (out_type, '*'):
                return self.resolve_link(graph, inp['link'], seen)
        return None

    def resolve_get_node(self, graph: Graph, node, seen):
        name = (node.get('widgets_values') or [None])[0]
        for other in graph.nodes.values():
            if other['type'] == 'SetNode' and (other.get('widgets_values') or [None])[0] == name:
                inputs = other.get('inputs', [])
                return self.resolve_link(graph, inputs[0].get('link'), seen) if inputs else None
        return None

    def inner_graph(self, graph: Graph, instance: dict) -> Graph:
        return Graph(self.subgraphs[instance['type']], graph.prefix + str(instance['id']) + ':', graph, instance)

    # --- API-Format ---

    def to_api(self, randomize=True) -> dict:
        """
         :param randomize: Seeds mit control_after_generate=randomize zufaellig setzen (wie das ComfyUI-Frontend)
        """
        self.warnings = []
        res = {}
        self._emit(Graph(self.workflow), res, randomize)
        return res

    def _emit(self, graph: Graph, res: dict, randomize):
        for node in graph.nodes.values():
            if node.get('mode', 0) in (MODE_MUTED, MODE_BYPASS):
                continue
            if node['type'] in self.subgraphs:
                self._emit(self.inner_graph(graph, node), res, randomize)
                continue
            info = self.object_info.get(node['type'])
            if info is None:
                continue  # Notizen, Reroute, Get/Set, PrimitiveNode und unbekannte Frontend-Nodes
            api_id = graph.prefix + str(node['id'])
            values = node.get('widgets_values')
            layout = self.layout(node)
            if isinstance(values, dict):
                inputs = dict(values)
            else:
                values = values or []
                inputs = {}
                for name, pos in layout.index.items():
                    if pos < len(values):
                        inputs[name] = values[pos]
                    else:  # Widget ist neuer als der gespeicherte Workflow
                        default = default_value(layout.specs[name])
                        if default is not None:
                            inputs[name] = default
                if randomize:
                    for name, pos in layout.controls.items():
                        if pos < len(values) and values[pos] == 'randomize' and isinstance(inputs.get(name), int):
                            inputs[name] = random.randint(0, min(SEED_MAX, spec_options(layout.specs[name]).get('max', SEED_MAX)))
            for inp in node.get('inputs', []):
                if inp.get('link') is None:
                    continue
                src = self.resolve_link(graph, inp['link'])
                if src is None:
                    continue  # nicht aufloesbar: Widget-Wert bleibt
                inputs[inp['name']] = [src[1], src[2]] if src[0] == 'node' else src[1]
            title = node.get('title') or info.get('display_name') or node['type']
            res[api_id] = {'class_type': node['type'], 'inputs': inputs, '_meta': {'title': title}}

    def validate(self, api: dict) -> list:
        """ :return: Hinweise zu fehlenden Pflicht-Eingaengen und ungueltigen Auswahlwerten """
        res = list(self.warnings)
        for api_id, node in api.items():
            info = self.object_info.get(node['class_type'], {})
            for name, spec in (info.get('input', {}).get('required') or {}).items():
                grown = any(key.startswith(name + '.') for key in node['inputs'])  # COMFY_AUTOGROW_V3: values.a, ...
                if name not in node['inputs'] and not grown and not spec_options(spec).get('socketless'):
                    res.append(f"Node {api_id} ({node['_meta']['title']}): Eingang {name} fehlt")
            for name, value in node['inputs'].items():
                spec = (info.get('input', {}).get('required') or {}).get(name) or \
                       (info.get('input', {}).get('optional') or {}).get(name)
                choices = spec_choices(spec) if spec is not None and spec_type(spec) == 'COMBO' else []
                if choices and not isinstance(value, list) and value not in choices and \
                        all(isinstance(c, str) for c in choices):
                    res.append(f"Node {api_id} ({node['_meta']['title']}): {name}={value} ist keine gueltige Auswahl")
        return res

    # --- Parameter ---

    def subgraph_input_spec(self, instance: dict, name: str):
        """ Spezifikation des ersten inneren Eingangs, an dem ein Subgraph-Eingang haengt """
        sub = self.subgraphs[instance['type']]
        inner = Graph(sub)
        for sub_input in sub.get('inputs', []):
            if sub_input.get('name') != name:
                continue
            for link_id in sub_input.get('linkIds', []):
                link = inner.links.get(link_id)
                target = inner.nodes.get(link[2]) if link else None
                if target is None:
                    continue
                inputs = target.get('inputs', [])
                if link[3] >= len(inputs):
                    continue
                target_name = inputs[link[3]].get('name')
                if target['type'] in self.subgraphs:
                    return self.subgraph_input_spec(target, target_name)
                info = self.object_info.get(target['type'], {})
                spec = self.layout(target).specs.get(target_name)
                if spec is None:
                    spec = (info.get('input', {}).get('required') or {}).get(target_name) or \
                           (info.get('input', {}).get('optional') or {}).get(target_name)
                if spec is not None:
                    return spec
        return None

    def params(self) -> list:
        """
         Wichtige Parameter: die nach aussen gefuehrten Widgets der Subgraphen und Widgets bestimmter Nodes
         (IMPORTANT_NODES, Empty*Latent*), jeweils nur auf oberster Ebene und nicht verbunden.

         :return: [{node, name, label, group, type, value, choices, options, seed, randomize}]
        """
        res = []
        for node in self.workflow.get('nodes', []):
            if node.get('mode', 0) in (MODE_MUTED, MODE_BYPASS):
                continue
            linked = {i['name'] for i in node.get('inputs', []) if i.get('link') is not None}
            values = self.widget_values(node)
            node_type = node['type']
            if node_type in self.subgraphs:
                group = node.get('title') or self.subgraphs[node_type].get('name') or node_type
                for inp in self.instance_widget_inputs(node):
                    if inp['name'] in linked or inp['name'] not in values:
                        continue
                    spec = self.subgraph_input_spec(node, inp['name'])
                    res.append(self._param(node, inp['name'], inp.get('label') or inp['name'], group,
                                           spec, inp.get('type'), values[inp['name']], None))
                continue
            if node_type in IMPORTANT_NODES:
                names = IMPORTANT_NODES[node_type]
            elif node_type.startswith('Empty') and 'Latent' in node_type:
                names = LATENT_PARAMS
            else:
                continue
            info = self.object_info.get(node_type, {})
            group = node.get('title') or info.get('display_name') or node_type
            if node_type == 'PrimitiveNode':
                outputs = node.get('outputs', [])
                if 'value' in values:
                    res.append(self._param(node, 'value', 'value', group, None,
                                           outputs[0].get('type') if outputs else None, values['value'], None))
                continue
            layout = self.layout(node)
            node_values = node.get('widgets_values') if isinstance(node.get('widgets_values'), list) else []
            for name in (names if names is not None else list(layout.index.keys())):
                if name in linked or name not in values:
                    continue
                control_pos = layout.controls.get(name)
                control = node_values[control_pos] if control_pos is not None and control_pos < len(node_values) else None
                res.append(self._param(node, name, name, group, layout.specs.get(name), None, values[name], control))
        return res

    @staticmethod
    def _param(node, name, label, group, spec, datatype, value, control) -> dict:
        datatype = spec_type(spec) if spec is not None else datatype
        if datatype in (None, '*'):
            datatype = {bool: 'BOOLEAN', int: 'INT', float: 'FLOAT'}.get(type(value), 'STRING')
        seed = datatype == 'INT' and 'seed' in name.lower()
        options = spec_options(spec)
        return {
            'node': str(node['id']), 'name': name, 'label': label, 'group': group, 'type': datatype,
            'value': value, 'choices': spec_choices(spec) if datatype == 'COMBO' and spec is not None else [],
            'options': {k: options[k] for k in ('min', 'max', 'step', 'multiline', 'tooltip') if k in options},
            'seed': seed, 'randomize': seed and control in (None, 'randomize'),
        }

    def set_control(self, node_id, name, value) -> bool:
        """ control_after_generate eines Widgets setzen (randomize, fixed, ...), falls der Node eins hat """
        node = next((n for n in self.workflow.get('nodes', []) if str(n['id']) == str(node_id)), None)
        values = node.get('widgets_values') if node is not None else None
        if node is None or node['type'] in self.subgraphs or not isinstance(values, list):
            return False
        pos = self.layout(node).controls.get(name)
        if pos is None or pos >= len(values):
            return False
        values[pos] = value
        return True

    def set_param(self, node_id, name, value) -> bool:
        node = next((n for n in self.workflow.get('nodes', []) if str(n['id']) == str(node_id)), None)
        if node is None:
            return False
        values = node.get('widgets_values')
        if isinstance(values, dict):
            values[name] = value
            return True
        pos = self.widget_index(node, name)
        if pos is None or not isinstance(values, list) or pos >= len(values):
            return False
        values[pos] = value
        return True
