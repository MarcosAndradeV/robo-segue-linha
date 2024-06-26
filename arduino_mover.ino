/****************
 * This sketch simulates a driving scenerio 
 * created 14 Nov 2018
 * by Ajala E Oladapo
 * @ InvenTech inc.
 * 
 * L298N takes a minimum of 12V and also gives out
 * 5V which can be used to power the InventOne brd.
 * We don't need an enable pin here because we assume
 * the two jumpers are plugged in.
****************/

#include <Drive.h>  //Include the Drive library

//Define L298N pin mappings
const int IN1 = 8;
const int IN2 = 9;
const int IN3 = 10;
const int IN4 = 11;

Drive drive(IN1, IN2, IN3, IN4);  //Create an instance of the function

void setup() {
//Begin serial communication with monitor
  Serial.begin(9600);
  
  Serial.println("Welcome to smart world");
}

void loop() {
  drive.moveForward(500);
  delay(3000);
  drive.moveBackward(500);
  delay(3000);
  drive.turnRight(500);
  delay(3000);
  drive.turnLeft(500);
  delay(3000);
}
