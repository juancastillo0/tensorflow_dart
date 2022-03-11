import 'dart:html';

import 'package:bootstrap_dart/bootstrap/bootstrap_core.dart';
import 'package:bootstrap_dart/bootstrap/form.dart';
import 'package:bootstrap_dart/mobx_deact.dart';
import 'package:deact/deact.dart';
import 'package:deact/deact_html52.dart';
import 'package:bootstrap_dart/bootstrap/checks_radios.dart';
import 'package:tensorflow_wasm/models/hand_pose.dart';

import '../shared/params.dart';

Future<void> setupDatGui(UrlSearchParams params) {
  final controller = deact(
    '#output',
    (_) => div(
      children: [
        div(id: 'stats'),
        div(
          id: 'main',
          children: [
            div(
              className: 'canvas-wrapper',
              children: [
                canvas(id: CANVAS_ELEMENT_ID),
                el(
                  'video',
                  attributes: {
                    'id': 'video',
                    'playsinline': true,
                    'style': '''
                            -webkit-transform: scaleX(-1);
                            transform: scaleX(-1);
                            visibility: hidden;
                            width: auto;
                            height: auto;''',
                  },
                )
              ],
            ),
          ],
        ),
        const OptionPanel(),
      ],
    ),
    wrappers: const [mobxWrapper],
  );

  return controller.waitScheduledRender();
}

class OptionPanel extends ComponentNode {
  const OptionPanel() : super();

  DeactNode render(ctx) {
    return div(
      children: [
        RadiosInput(
          name: 'model-type',
          onChanged: (v) {
            STATE.modelConfig.type = MediaPipeHandsModelType.values.byName(v);
            STATE.isModelChanged = true;
          },
          items: Map.fromIterable(
            MediaPipeHandsModelType.values.map((e) => e.name),
            value: (v) => txt(v as String),
          ),
          selectedId: STATE.modelConfig.type?.name,
        ),
        labeledInput(
          wrapperDivClass: 'my-2',
          label: txt('Size Option'),
          id: 'size-option',
          divClass: 'row',
          // colClasses: ColInputClasses(),
          input: _simpleSelect<String>(
            VIDEO_SIZE.keys.toList(),
            (d) => d,
            STATE.camera.sizeOption.value,
            (sizeOption) {
              STATE.camera.sizeOption.value = sizeOption;
              STATE.isSizeOptionChanged = true;
            },
            id: 'size-option',
          ),
        ),
        div(
          style: colStyle(),
          children: [
            div(
              style: flexStyle(
                main: AxisAlign.space_between,
                expand: true,
              ),
              children: [
                label(
                  forId: 'max-num-hands',
                  children: [
                    txt('maxNumHands'),
                  ],
                ),
              ],
            ),
            input(
              type: 'range',
              value: STATE.modelConfig.maxNumHands?.toString(),
              min: '1',
              max: '10',
              step: '1',
              id: 'max-num-hands',
              // className: formRangeClass,
              oninput: (e) {
                final value = int.parse((e.target as InputElement).value!);
                STATE.modelConfig.maxNumHands = value;
                STATE.isModelChanged = true;
              },
            ),
          ],
        )
      ],
    );
  }
}

DeactNode _simpleSelect<T>(
  List<T> values,
  String Function(T) toString,
  T state,
  void Function(T) onChange, {
  String? id,
}) {
  return select(
    id: id,
    className: 'form-select mx-1',
    style: 'width:170px;',
    onchange: (e) {
      final value = (e.target! as SelectElement).value;
      onChange(values.firstWhere((v) => toString(v) == value));
    },
    children: [
      ...values.map(
        (e) => option(
          value: toString(e),
          selected: state == e ? '' : null,
          children: [txt(toString(e))],
        ),
      )
    ],
  );
}
