import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'providers/session_provider.dart';
import 'ui/screens/live_session_screen.dart';
import 'ui/screens/setup_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ConferenceApp());
}

class ConferenceApp extends StatelessWidget {
  const ConferenceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => SessionProvider(),
      child: MaterialApp(
        title: 'Conference Translator',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorSchemeSeed: Colors.indigo,
          useMaterial3: true,
        ),
        home: const _Root(),
      ),
    );
  }
}

class _Root extends StatefulWidget {
  const _Root();

  @override
  State<_Root> createState() => _RootState();
}

class _RootState extends State<_Root> {
  bool _inSession = false;

  @override
  void initState() {
    super.initState();
    WakelockPlus.enable();
  }

  @override
  Widget build(BuildContext context) {
    if (_inSession) {
      return LiveSessionScreen(onBack: () => setState(() => _inSession = false));
    }
    return SetupScreen(onReady: () => setState(() => _inSession = true));
  }
}
