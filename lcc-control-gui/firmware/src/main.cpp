#include <Arduino.h>

struct Position {
  float x;
  float y;
  float z;
};

struct MachineState {
  Position current;
  Position target;
  bool absoluteMode;
  bool isHomed;
  bool isMoving;
  float feedrate;
  float acceleration;
  float maxFeedrateX;
  float maxFeedrateY;
  float maxFeedrateZ;
  unsigned long moveStartTime;
  unsigned long moveDuration;
};

MachineState state = {
  {0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0},
  true,
  false,
  false,
  1000.0,
  500.0,
  5000.0,
  5000.0,
  5000.0,
  0,
  0
};

void handleCommand(String cmd);
void updateMovement();
float parseValue(String cmd, char axis);
bool hasParameter(String cmd, char param);
float calculateDistance(Position from, Position to);
unsigned long calculateMoveDuration(float distance, float feedrate);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(1000);

  Serial.println("start");
  Serial.println("Marlin 2.0.0");
  Serial.println("echo:  Last Updated: 2024-01-01 12:00");
  Serial.println("echo: Compiled: Jan  1 2024");
  Serial.println("echo: Free Memory: 32768");
  Serial.println("echo:Hardcoded Default Settings Loaded");
}

void loop() {
  updateMovement();

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() > 0) {
      handleCommand(cmd);
    }
  }
}

void updateMovement() {
  if (!state.isMoving) return;

  unsigned long elapsed = millis() - state.moveStartTime;
  if (elapsed >= state.moveDuration) {
    state.current = state.target;
    state.isMoving = false;
  } else {
    float progress = (float)elapsed / state.moveDuration;
    state.current.x = state.current.x + (state.target.x - state.current.x) * progress;
    state.current.y = state.current.y + (state.target.y - state.current.y) * progress;
    state.current.z = state.current.z + (state.target.z - state.current.z) * progress;
  }
}

float parseValue(String cmd, char param) {
  int idx = cmd.indexOf(param);
  if (idx == -1) return 0.0;
  return cmd.substring(idx + 1).toFloat();
}

bool hasParameter(String cmd, char param) {
  return cmd.indexOf(param) != -1;
}

float calculateDistance(Position from, Position to) {
  float dx = to.x - from.x;
  float dy = to.y - from.y;
  float dz = to.z - from.z;
  return sqrt(dx*dx + dy*dy + dz*dz);
}

unsigned long calculateMoveDuration(float distance, float feedrate) {
  if (feedrate <= 0) feedrate = 1000.0;
  return (unsigned long)((distance / feedrate) * 60000.0);
}

void handleCommand(String cmd) {
  delay(10);

  if (cmd.startsWith("M112")) {
    Serial.println("echo:EMERGENCY STOP");
    state.isMoving = false;
    Serial.println("ok");
  }
  else if (cmd.startsWith("G28")) {
    Serial.println("echo:Homing");

    bool homeX = !hasParameter(cmd, 'X') && !hasParameter(cmd, 'Y') && !hasParameter(cmd, 'Z');
    bool homeY = homeX;
    bool homeZ = homeX;

    if (!homeX) {
      homeX = hasParameter(cmd, 'X');
      homeY = hasParameter(cmd, 'Y');
      homeZ = hasParameter(cmd, 'Z');
    }

    delay(500);

    if (homeX) {
      state.current.x = 0.0;
      state.target.x = 0.0;
      Serial.println("echo:  Home X");
    }
    if (homeY) {
      state.current.y = 0.0;
      state.target.y = 0.0;
      Serial.println("echo:  Home Y");
    }
    if (homeZ) {
      state.current.z = 0.0;
      state.target.z = 0.0;
      Serial.println("echo:  Home Z");
    }

    state.isHomed = true;
    state.isMoving = false;
    Serial.println("ok");
  }
  else if (cmd.startsWith("M114")) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer),
             "X:%.2f Y:%.2f Z:%.2f E:0.00 Count X:%.2f Y:%.2f Z:%.2f",
             state.current.x, state.current.y, state.current.z,
             state.current.x, state.current.y, state.current.z);
    Serial.println(buffer);
    Serial.println("ok");
  }
  else if (cmd.startsWith("G90")) {
    state.absoluteMode = true;
    Serial.println("ok");
  }
  else if (cmd.startsWith("G91")) {
    state.absoluteMode = false;
    Serial.println("ok");
  }
  else if (cmd.startsWith("G0") || cmd.startsWith("G1")) {
    Position newTarget = state.current;

    if (hasParameter(cmd, 'X')) {
      float x = parseValue(cmd, 'X');
      newTarget.x = state.absoluteMode ? x : state.current.x + x;
    }
    if (hasParameter(cmd, 'Y')) {
      float y = parseValue(cmd, 'Y');
      newTarget.y = state.absoluteMode ? y : state.current.y + y;
    }
    if (hasParameter(cmd, 'Z')) {
      float z = parseValue(cmd, 'Z');
      newTarget.z = state.absoluteMode ? z : state.current.z + z;
    }
    if (hasParameter(cmd, 'F')) {
      state.feedrate = parseValue(cmd, 'F');
    }

    float distance = calculateDistance(state.current, newTarget);
    if (distance > 0.001) {
      state.target = newTarget;
      state.isMoving = true;
      state.moveStartTime = millis();
      state.moveDuration = calculateMoveDuration(distance, state.feedrate);
    }

    Serial.println("ok");
  }
  else if (cmd.startsWith("M204")) {
    if (hasParameter(cmd, 'P')) {
      state.acceleration = parseValue(cmd, 'P');
      Serial.print("echo:Acceleration set to ");
      Serial.println(state.acceleration);
    }
    Serial.println("ok");
  }
  else if (cmd.startsWith("M203")) {
    if (hasParameter(cmd, 'X')) {
      state.maxFeedrateX = parseValue(cmd, 'X');
    }
    if (hasParameter(cmd, 'Y')) {
      state.maxFeedrateY = parseValue(cmd, 'Y');
    }
    if (hasParameter(cmd, 'Z')) {
      state.maxFeedrateZ = parseValue(cmd, 'Z');
    }
    Serial.println("echo:Max feedrate updated");
    Serial.println("ok");
  }
  else if (cmd.startsWith("M42")) {
    int pin = 67;
    int value = 0;

    if (hasParameter(cmd, 'P')) {
      pin = (int)parseValue(cmd, 'P');
    }
    if (hasParameter(cmd, 'S')) {
      value = (int)parseValue(cmd, 'S');
    }

    Serial.print("echo:Pin ");
    Serial.print(pin);
    Serial.print(" set to ");
    Serial.println(value);
    Serial.println("ok");
  }
  else if (cmd.startsWith("M502")) {
    Serial.println("echo:Hard Reset");
    state.acceleration = 500.0;
    state.maxFeedrateX = 5000.0;
    state.maxFeedrateY = 5000.0;
    state.maxFeedrateZ = 5000.0;
    state.feedrate = 1000.0;
    Serial.println("ok");
  }
  else if (cmd.startsWith("M115")) {
    Serial.println("FIRMWARE_NAME:Marlin 2.0.0 (Simulator)");
    Serial.println("FIRMWARE_URL:https://github.com");
    Serial.println("PROTOCOL_VERSION:1.0");
    Serial.println("MACHINE_TYPE:3D Printer");
    Serial.println("EXTRUDER_COUNT:0");
    Serial.println("UUID:00000000-0000-0000-0000-000000000000");
    Serial.println("ok");
  }
  else if (cmd.startsWith("G4")) {
    int ms = 0;
    if (hasParameter(cmd, 'P')) {
      ms = (int)parseValue(cmd, 'P');
    } else if (hasParameter(cmd, 'S')) {
      ms = (int)(parseValue(cmd, 'S') * 1000);
    }

    Serial.print("echo:Dwell ");
    Serial.print(ms);
    Serial.println("ms");
    delay(min(ms, 5000));
    Serial.println("ok");
  }
  else {
    Serial.println("ok");
  }
}