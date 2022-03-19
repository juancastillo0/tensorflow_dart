import 'dart:html';

import 'package:bootstrap_dart/bootstrap/bootstrap_core.dart';
import 'package:bootstrap_dart/mobx_deact.dart';
import 'package:deact/deact.dart';
import 'package:deact/deact_html52.dart';
import 'package:bootstrap_dart/bootstrap/checks_radios.dart';

import 'main.dart';

Future<void> setupFaceLandmarkGui(FaceLandmarkState state) {
  final controller = deact(
    '#output',
    (ctx) {
      return div(
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
          OptionPanel(state),
        ],
      );
    },
    wrappers: const [mobxWrapper],
  );

  return controller.waitScheduledRender();
}

class OptionPanel extends ComponentNode {
  const OptionPanel(this.state) : super();
  final FaceLandmarkState state;

  DeactNode render(ctx) {
    return div(
      children: [
        // RadiosInput(
        //   name: 'model-type',
        //   onChanged: (v) {
        //     state.modelConfig.type = MediaPipeHandsModelType.values.byName(v);
        //     state.isModelChanged = true;
        //   },
        //   items: Map.fromIterable(
        //     MediaPipeHandsModelType.values.map((e) => e.name),
        //     value: (v) => txt(v as String),
        //   ),
        //   selectedId: state.modelConfig.type?.name,
        // ),
        // labeledInput(
        //   wrapperDivClass: 'my-2',
        //   label: txt('Size Option'),
        //   id: 'size-option',
        //   divClass: 'row',
        //   // colClasses: ColInputClasses(),
        //   input: _simpleSelect<String>(
        //     VIDEO_SIZE.keys.toList(),
        //     (d) => d,
        //     state.camera.sizeOption.value,
        //     (sizeOption) {
        //       state.camera.sizeOption.value = sizeOption;
        //     },
        //     id: 'size-option',
        //   ),
        // ),
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
                  forId: 'max-num-faces',
                  children: [
                    txt('maxNumFaces'),
                  ],
                ),
              ],
            ),
            input(
              type: 'range',
              value: state.maxFaces.value.toString(),
              min: '1',
              max: '20',
              step: '1',
              id: 'max-num-faces',
              // className: formRangeClass,
              oninput: (e) {
                final value = int.parse((e.target as InputElement).value!);
                state.maxFaces.value = value;
              },
            ),
          ],
        ),
        check(
          divClass: 'mt-2',
          checked: state.predictIrises.value,
          onChange: (v) => state.predictIrises.value = v,
          id: 'predict-irises',
          label: txt('Predict Irises'),
          type: CheckType.checkbox,
        ),
        check(
          divClass: 'mt-2',
          checked: state.triangulateMesh.value,
          onChange: (v) => state.triangulateMesh.value = v,
          id: 'triangulate-mesh',
          label: txt('Triangulate Mesh'),
          type: CheckType.checkbox,
        ),
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
