// import 'dart:async';
// import 'dart:io';
// import 'package:camera/camera.dart';
// import 'package:flutter/foundation.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter_dotenv/flutter_dotenv.dart';
// import 'package:web_socket_channel/web_socket_channel.dart';
//
// class LiveAnalysisScreen extends StatefulWidget {
//   const LiveAnalysisScreen({super.key});
//
//   @override
//   State<LiveAnalysisScreen> createState() => _LiveAnalysisScreenState();
// }
//
// class _LiveAnalysisScreenState extends State<LiveAnalysisScreen> {
//   CameraController? _cameraController;
//   WebSocketChannel? _channel;
//   String _analysisResult = 'Connecting...';
//   bool _isStreaming = false;
//   Timer? _streamTimer;
//
//   final String _prompt = "What is happening in this scene?"; // Or get from user
//
//   @override
//   void initState() {
//     super.initState();
//     _initializeCameraAndConnection();
//   }
//
//   Future<void> _initializeCameraAndConnection() async {
//     try {
//       final cameras = await availableCameras();
//       if (cameras.isEmpty) {
//         throw Exception('No cameras found');
//       }
//       final firstCamera = cameras.first;
//
//       _cameraController = CameraController(
//         firstCamera,
//         ResolutionPreset.medium,
//         enableAudio: false,
//       );
//
//       await _cameraController!.initialize();
//       _connectToWebSocket();
//
//       setState(() {});
//     } catch (e) {
//       setState(() {
//         _analysisResult = "Error initializing camera: $e";
//       });
//     }
//   }
//
//   void _connectToWebSocket() {
//     final wsUrl = (kIsWeb
//         ? dotenv.env['SERVER_URL_WEB']
//         : Platform.isAndroid
//         ? dotenv.env['SERVER_URL_ANDROID']
//         : dotenv.env['SERVER_URL_IOS'])!
//         .replaceFirst('http', 'ws') // Change http to ws for WebSocket
//         .replaceAll('/analyze-image/', '/ws/live-analysis');
//
//     _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
//     _channel!.sink.add(_prompt); // Send the initial prompt
//
//     _channel!.stream.listen((message) {
//       setState(() {
//         _analysisResult = message;
//       });
//     }, onError: (error) {
//       setState(() {
//         _analysisResult = "Connection Error: $error";
//       });
//       _isStreaming = false;
//     }, onDone: () {
//       setState(() {
//         _analysisResult = "Connection Closed.";
//         _isStreaming = false;
//       });
//     });
//   }
//
//   void _toggleStreaming() {
//     if (_isStreaming) {
//       // Stop streaming
//       _streamTimer?.cancel();
//       setState(() {
//         _isStreaming = false;
//       });
//     } else {
//       // Start streaming
//       _streamTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
//         _sendFrame();
//       });
//       setState(() {
//         _isStreaming = true;
//       });
//     }
//   }
//
//   Future<void> _sendFrame() async {
//     if (_cameraController == null || !_cameraController!.value.isInitialized) {
//       return;
//     }
//     try {
//       final image = await _cameraController!.takePicture();
//       final imageBytes = await image.readAsBytes();
//       _channel?.sink.add(imageBytes);
//     } catch (e) {
//       debugPrint("Error sending frame: $e");
//     }
//   }
//
//   @override
//   void dispose() {
//     _streamTimer?.cancel();
//     _cameraController?.dispose();
//     _channel?.sink.close();
//     super.dispose();
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: const Text('Live Analysis')),
//       body: Column(
//         children: [
//           Expanded(
//             child: _cameraController == null ||
//                 !_cameraController!.value.isInitialized
//                 ? const Center(child: CircularProgressIndicator())
//                 : Container(
//               margin: const EdgeInsets.all(8.0),
//               decoration: BoxDecoration(
//                 border: Border.all(color: Colors.teal, width: 3),
//                 borderRadius: BorderRadius.circular(12),
//               ),
//               child: ClipRRect(
//                 borderRadius: BorderRadius.circular(9),
//                 child: CameraPreview(_cameraController!),
//               ),
//             ),
//           ),
//           Padding(
//             padding: const EdgeInsets.all(16.0),
//             child: Card(
//               elevation: 4,
//               child: Container(
//                 width: double.infinity,
//                 padding: const EdgeInsets.all(16.0),
//                 child: Text(
//                   _analysisResult,
//                   style: Theme.of(context).textTheme.bodyLarge,
//                   textAlign: TextAlign.center,
//                 ),
//               ),
//             ),
//           ),
//         ],
//       ),
//       floatingActionButton: FloatingActionButton(
//         onPressed: _toggleStreaming,
//         child: Icon(_isStreaming ? Icons.stop : Icons.play_arrow),
//       ),
//       floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
//     );
//   }
// }