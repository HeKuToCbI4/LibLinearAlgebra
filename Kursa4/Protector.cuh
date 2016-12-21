#pragma once
#include <Windows.h>

class Protector
{
	__declspec(dllexport) static HKEY OpenKey(HKEY hRootKey, LPCSTR strKey);
	__declspec(dllexport) void SetVal(LPCTSTR lpValue, DWORD data) const;
	__declspec(dllexport) DWORD GetVal(LPCTSTR lpValue) const;
	unsigned launch_count;
	unsigned hash_val;
	__declspec(dllexport) static unsigned hash(unsigned x); HKEY hKey;
	static Protector* instance;
	Protector();
public:
	__declspec(dllexport) static Protector* get_instance();
	__declspec(dllexport)  bool validate() const;
	__declspec(dllexport) ~Protector();
};

