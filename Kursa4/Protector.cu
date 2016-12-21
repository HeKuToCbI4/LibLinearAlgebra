#include "Protector.cuh"
#include <sstream>
#include <string>
#include <Windows.h>
#include <iostream>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define TRIAL_LAUNCHES 25

using std::cout; using std::endl;

Protector* Protector::instance;

HKEY Protector::OpenKey(HKEY hRootKey, LPCSTR strKey)
{
	HKEY hKey;
	LONG nError = RegOpenKeyEx(hRootKey, strKey, NULL, KEY_ALL_ACCESS, &hKey);

	if (nError == ERROR_FILE_NOT_FOUND)
	{
		cout << "Creating registry key: " << strKey << endl;
		nError = RegCreateKeyEx(hRootKey, strKey, NULL, nullptr, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, nullptr, &hKey, nullptr);
	}

	if (nError)
		cout << "Error: " << nError << " Could not find or create " << strKey << endl;

	return hKey;
}

void Protector::SetVal(LPCTSTR lpValue, DWORD data) const
{
	LONG nError = RegSetValueEx(hKey, lpValue, NULL, REG_DWORD, reinterpret_cast<LPBYTE>(&data), sizeof(DWORD));

	if (nError)
		cout << "Error: " << nError << " Could not set registry value: " << const_cast<char*>(lpValue) << endl;
}

DWORD Protector::GetVal(LPCTSTR lpValue) const
{
	DWORD data;		DWORD size = sizeof(data);	DWORD type = REG_DWORD;
	LONG nError = RegQueryValueEx(hKey, lpValue, nullptr, &type, reinterpret_cast<LPBYTE>(&data), &size);

	if (nError == ERROR_FILE_NOT_FOUND)
		data = 0; // The value will be created and set to data next time SetVal() is called.
	else if (nError)
		cout << "Error: " << nError << " Could not get registry value " << const_cast<char*>(lpValue) << endl;

	return data;
}

//unsigned get_system_drive_id()
//{
//	DWORD dwSerial;
//	std::stringstream ss;
//
//	if (!GetVolumeInformation(TEXT("C:\\"), NULL, 0, &dwSerial, NULL, NULL, NULL, 0)) {
//		ss << "Error: " << GetLastError();
//	}
//	else {
//		ss << dwSerial;
//	}
//	return stoi(ss.str());
//}

unsigned Protector::hash(unsigned x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}

Protector::Protector()
{
	hKey = OpenKey(HKEY_CURRENT_USER, "LibLinearAlgebra\\Leonov");
	if (GetVal("Launch_count") == 0)
	{
		SetVal("Launch_count", 0);
	}
	launch_count = GetVal("Launch_count");
	if (!validate())
		throw std::exception("TRIAL HAD EXPIRED!");
	launch_count++;
	SetVal("Launch_count", launch_count);

}

Protector* Protector::get_instance()
{
	if (instance == nullptr)
		instance = new Protector();
	return instance;

}

bool Protector::validate() const
{
	if (TRIAL_LAUNCHES == launch_count)
		return false;
	return true;
}

Protector::~Protector()
{
	RegCloseKey(hKey);
}
