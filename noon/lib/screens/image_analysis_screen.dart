import 'dart:io';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';

class ImageAnalysisScreen extends StatefulWidget {
  const ImageAnalysisScreen({super.key});

  @override
  State<ImageAnalysisScreen> createState() => _ImageAnalysisScreenState();
}

class _ImageAnalysisScreenState extends State<ImageAnalysisScreen> {
  XFile? _imageFile;
  Map<String, dynamic>? _analysisResult;
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();
  final ApiService _apiService = ApiService();

  // Function to reset the state to the initial view
  void _resetState() {
    setState(() {
      _imageFile = null;
      _analysisResult = null;
      _isLoading = false;
    });
  }

  void _showImageSourceActionSheet(BuildContext context) {
    if (kIsWeb) {
      _pickImage(ImageSource.gallery);
      return;
    }
    showModalBottomSheet(
      context: context,
      builder: (context) => SafeArea(
        child: Wrap(
          children: [
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('Photo Library'),
              onTap: () {
                _pickImage(ImageSource.gallery);
                Navigator.of(context).pop();
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo_camera),
              title: const Text('Camera'),
              onTap: () {
                _pickImage(ImageSource.camera);
                Navigator.of(context).pop();
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _imageFile = pickedFile;
          _analysisResult = null; // Clear previous result
        });
      }
    } catch (e) {
      _showErrorDialog("Failed to pick image: $e");
    }
  }

  Future<void> _analyzeImage() async {
    if (_imageFile == null) {
      _showErrorDialog("Please select an image first.");
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final result = await _apiService.analyzeImage(_imageFile!);
      setState(() {
        _analysisResult = result;
      });
    } catch (e) {
      _showErrorDialog(e.toString());
    } finally {
      if(mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showErrorDialog(String message) {
    if (!mounted) return;
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('An Error Occurred'),
        content: Text(message),
        actions: <Widget>[
          TextButton(
            child: const Text('Okay'),
            onPressed: () => Navigator.of(ctx).pop(),
          )
        ],
      ),
    );
  }

  Widget _buildResultsCard(Map<String, dynamic> results) {
    if (results.containsKey('error')) {
      return Text(
        "Error from model: ${results['error']}\n\nRaw Response: ${results['raw_response']}",
        style: const TextStyle(color: Colors.red),
      );
    }

    final String foodItem = results['food_item']?.toString() ?? 'N/A';

    int calories = 0; // Default value
    if (results.containsKey('calories') &&
        results['calories'] != null &&
        results['calories'] != 'N/A') {
        calories = int.tryParse(results['calories'].toString()) ?? 0;
      }

    final Map<String, dynamic> ingredients = (results['ingredients'] is Map)
        ? Map<String, dynamic>.from(results['ingredients'])
        : {};

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          foodItem,
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        Text(
          '$calories Calories (Estimated)',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey[700]),
        ),
        const Divider(height: 24),
        Text(
          'Nutritional Info:',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 12),
        ...ingredients.entries.map((entry) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '${entry.key[0].toUpperCase()}${entry.key.substring(1)}'.replaceAll('_', ' '),
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                Text(entry.value.toString()),
              ],
            ),
          );
        }).toList(),
      ],
    );
  }

  // --- MODIFIED: The action button logic is updated for better UX ---
  Widget _buildActionButtons() {
    // State 1: No image is selected yet. Show a single "Select Image" button.
    if (_imageFile == null) {
      return SizedBox(
        width: double.infinity,
        child: ElevatedButton.icon(
          icon: const Icon(Icons.image_search),
          label: const Text('Select Image'),
          onPressed: () => _showImageSourceActionSheet(context),
          style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
        ),
      );
    }
    // State 2: Analysis is complete. Show a single "Analyze Another Image" button.
    else if (_analysisResult != null && !_isLoading) {
      return SizedBox(
        width: double.infinity,
        child: ElevatedButton.icon(
          icon: const Icon(Icons.refresh),
          label: const Text('Analyze Another Image'),
          onPressed: _resetState,
          style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
        ),
      );
    }
    // State 3: Image is selected, but not yet analyzed. Show two options.
    else {
      return Row(
        children: [
          Expanded(
            child: OutlinedButton.icon(
              icon: const Icon(Icons.image_search),
              label: const Text('Reselect'),
              onPressed: _isLoading ? null : () => _showImageSourceActionSheet(context),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: ElevatedButton.icon(
              icon: const Icon(Icons.analytics_outlined),
              label: const Text('Analyze'),
              onPressed: _isLoading ? null : _analyzeImage,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
          ),
        ],
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Food Analyzer'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              // --- Image Display Area ---
              Container(
                width: double.infinity,
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey[700]!, width: 2),
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.black.withOpacity(0.2),
                ),
                child: _imageFile != null
                    ? ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: kIsWeb
                      ? Image.network(_imageFile!.path, fit: BoxFit.contain)
                      : Image.file(File(_imageFile!.path), fit: BoxFit.contain),
                )
                    : const Center(
                  child: Text('Select an image to analyze', style: TextStyle(color: Colors.grey)),
                ),
              ),
              const SizedBox(height: 20),

              // --- MODIFIED: Use the new button logic ---
              _buildActionButtons(),
              const SizedBox(height: 20),

              if (_isLoading)
                const Padding(
                  padding: EdgeInsets.all(16.0),
                  child: CircularProgressIndicator(),
                )
              else if (_analysisResult != null)
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    width: double.infinity,
                    child: _buildResultsCard(_analysisResult!),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}