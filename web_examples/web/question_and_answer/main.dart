import 'package:bootstrap_dart/bootstrap/bootstrap_core.dart';
import 'package:bootstrap_dart/bootstrap/form.dart';
import 'package:bootstrap_dart/hooks.dart';
import 'package:bootstrap_dart/mobx_deact.dart';
import 'package:deact/deact.dart';
import 'package:deact/deact_html52.dart';
import 'package:mobx/mobx.dart';

import 'package:tensorflow_wasm/models/question_and_answer.dart' as qa;
import 'package:universal_html/html.dart';

class _State {
  qa.QuestionAndAnswer? model;

  _State() {
    _load();
  }

  void _load() async {
    if (loadingModel.value) return;
    loadingModel.value = true;
    try {
      this.model = await qa.load(
          //   qa.ModelConfig(
          //   fromTFHub: false,
          //   modelUrl: '/question_and_answer/model/model.json',
          // )
          );
    } catch (e, s) {
      error.value = 'Error loading model: $e $s';
    } finally {
      loadingModel.value = false;
    }
  }

  final error = Observable('');
  final loadingModel = Observable(false);
  final loadingInference = Observable<DateTime?>(null);
  final inferenceMillis = Observable<int?>(null);
  final documentText = Observable(_teslaDocument);
  final questionText = Observable('');
  final answers = Observable(<qa.Answer>[]);

  void answerQuestion() async {
    if (model == null || loadingInference.value != null) return;
    loadingInference.value = DateTime.now();
    try {
      print(questionText.value);
      print(documentText.value);
      final answers = await model!.findAnswers(
        questionText.value,
        documentText.value,
      );
      print(answers);
      this.answers.value = answers;
    } finally {
      inferenceMillis.value =
          DateTime.now().difference(loadingInference.value!).inMilliseconds;
      loadingInference.value = null;
    }
  }
}

Future<void> main() {
  final controller = deact(
    '#output',
    (_) {
      return fc(
        (ctx) {
          final state = useMemo(ctx, () => _State());

          return fragment([
            div(id: 'stats'),
            div(
              id: 'main',
              className: 'container',
              children: [
                labeledInput(
                  label: txt('Document'),
                  id: 'question-document',
                  wrapperDivClass: 'my-2',
                  input: el(
                    'textarea',
                    attributes: {
                      'class': formControlClass(size: BSize.lg),
                      'placeholder':
                          'The supporting document with context for answering the question',
                      'id': 'question-document',
                      'style': 'height:400px;',
                    },
                    listeners: {
                      'oninput': (e) {
                        state.documentText.value =
                            (e.target as TextAreaElement).value!;
                      },
                    },
                    children: [txt(state.documentText.value)],
                  ),
                ),
                labeledInput(
                  label: txt('Question'),
                  id: 'question',
                  wrapperDivClass: 'my-2',
                  input: input(
                    className: formControlClass(size: BSize.lg),
                    type: 'text',
                    placeholder: 'Write your question',
                    id: 'question',
                    value: state.questionText.value,
                    oninput: (e) {
                      state.questionText.value =
                          (e.target as InputElement).value!;
                    },
                  ),
                ),
                div(
                  children: [
                    if (state.error.value.isNotEmpty) txt(state.error.value)
                  ],
                ),
                div(
                  style:
                      'display:flex;flex-direction:row;justify-content:center;padding:10px;',
                  children: [
                    button(
                      // style:
                      //     'display:flex;flex-direction:row;justify-content:center;',
                      className: btn(outlined: true),
                      children: [
                        if (state.loadingModel.value) ...[
                          spinner(
                            size: BSize.sm,
                            ariaHidden: true,
                            className: 'me-2',
                          ),
                          txt('Loading model...'),
                        ] else if (state.loadingInference.value != null) ...[
                          spinner(
                            size: BSize.sm,
                            ariaHidden: true,
                            className: 'me-2',
                          ),
                          txt('Processing...'),
                        ] else
                          txt('Answer'),
                      ],
                      onclick: (_) {
                        state.answerQuestion();
                      },
                    ),
                  ],
                ),
                div(
                  id: 'answers',
                  children: [
                    span(
                      children: [
                        if (state.inferenceMillis.value != null)
                          txt('Time: ${state.inferenceMillis.value} ms')
                      ],
                    ),
                    ...state.answers.value.map(
                      (e) => div(
                        children: [
                          span(children: [txt(e.score.toString())]),
                          span(children: [txt(e.startIndex.toString())]),
                          span(children: [txt(e.endIndex.toString())]),
                          span(children: [txt(e.text)]),
                        ],
                      ),
                    )
                  ],
                ),
              ],
            ),
          ]);
        },
      );
    },
    wrappers: const [mobxWrapper],
  );

  return controller.waitScheduledRender();
}

const _teslaDocument = '''
Nikola Tesla (/ˈtɛslə/;[2] Serbo-Croatian: [nǐkola têsla]; Serbian Cyrillic: Никола Тесла;[a] 10
July 1856 – 7 January 1943) was a Serbian-American[4][5][6] inventor, electrical engineer, mechanical engineer,
and futurist who is best known for his contributions to the design of the modern alternating current (AC)
electricity supply system.[7] <br/>

Born and raised in the Austrian Empire, Tesla studied engineering and physics in the 1870s without receiving a
degree, and gained practical experience in the early 1880s working in telephony and at Continental Edison in the
new electric power industry. He emigrated in 1884 to the United States, where he would become a naturalized
citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own.
With the help of partners to finance and market his ideas, Tesla set up laboratories and companies in New York to
develop a range of electrical and mechanical devices. His alternating current (AC) induction motor and related
polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and
became the cornerstone of the polyphase system which that company would eventually market.<br/>

Attempting to develop inventions he could patent and market, Tesla conducted a range of experiments with
mechanical oscillators/generators, electrical discharge tubes, and early X-ray imaging. He also built a
wireless-controlled boat, one of the first ever exhibited. Tesla became well known as an inventor and would
demonstrate his achievements to celebrities and wealthy patrons at his lab, and was noted for his showmanship at
public lectures. Throughout the 1890s, Tesla pursued his ideas for wireless lighting and worldwide wireless
electric power distribution in his high-voltage, high-frequency power experiments in New York and Colorado
Springs. In 1893, he made pronouncements on the possibility of wireless communication with his devices. Tesla
tried to put these ideas to practical use in his unfinished Wardenclyffe Tower project, an intercontinental
wireless communication and power transmitter, but ran out of funding before he could complete it.[8]<br/>

After Wardenclyffe, Tesla experimented with a series of inventions in the 1910s and 1920s with varying degrees of
success. Having spent most of his money, Tesla lived in a series of New York hotels, leaving behind unpaid bills.
He died in New York City in January 1943.[9] Tesla's work fell into relative obscurity following his death, until
1960, when the General Conference on Weights and Measures named the SI unit of magnetic flux density the tesla in
his honor.[10] There has been a resurgence in popular interest in Tesla since the 1990s.[11]''';
