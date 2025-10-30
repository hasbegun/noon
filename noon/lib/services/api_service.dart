import 'dart:convert';
import 'dart:io' show Platform, File;
import 'package:flutter/foundation.dart' show kIsWeb, Uint8List, kDebugMode, debugPrint;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:image_picker/image_picker.dart';

class ApiService {
  late final String _baseUrl;

  static const int MAX_FILE_SIZE_BYTES = 1024 * 700;

  ApiService() {
    _initializeBaseUrl();
  }

  void _initializeBaseUrl() {
    if (kIsWeb) {
      _baseUrl = dotenv.env['SERVER_URL_WEB'] ?? "http://127.0.0.1:8000";
    } else if (Platform.isAndroid) {
      _baseUrl = dotenv.env['SERVER_URL_ANDROID'] ?? "http://10.0.2.2:8000";
    } else {
      _baseUrl = dotenv.env['SERVER_URL_IOS'] ?? "http://127.0.0.1:8000";
    }
  }

  Future<Uint8List> _compressImage(XFile imageFile) async {
    final fileBytes = await imageFile.readAsBytes();
    if (fileBytes.length <= MAX_FILE_SIZE_BYTES) {
      return fileBytes;
    }
    CompressFormat format = imageFile.path.toLowerCase().endsWith('.png') ? CompressFormat.png : CompressFormat.jpeg;
    Uint8List? compressedBytes = await FlutterImageCompress.compressWithList(
      fileBytes,
      minWidth: 1080,
      minHeight: 1080,
      quality: 80,
      format: format,
    );
    if (kDebugMode && compressedBytes.length > MAX_FILE_SIZE_BYTES) {
      debugPrint("Warning: Compressed image size (${compressedBytes.length} bytes) still exceeds target.");
    }
    return compressedBytes;
  }

  /// Analyzes the image and returns the structured JSON data as a Map.
  Future<Map<String, dynamic>> analyzeImage(XFile imageFile) async {
    var uri = Uri.parse("$_baseUrl/analyze-image/");
    try {
      // The request no longer sends 'prompt' or 'engine' fields.
      var request = http.MultipartRequest('POST', uri);

      final compressedBytes = await _compressImage(imageFile);

      request.files.add(http.MultipartFile.fromBytes(
        'image',
        compressedBytes,
        filename: "upload.jpg",
        contentType: MediaType('image', 'jpeg'),
      ));

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['analysis'] as Map<String, dynamic>;
      } else {
        throw Exception(
            "Error from server: ${response.statusCode}\nBody: ${response.body}");
      }
    } catch (e) {
      throw Exception(
          "Failed to connect to the server at $_baseUrl. Is it running?\n\nDetails: $e");
    }
  }
}