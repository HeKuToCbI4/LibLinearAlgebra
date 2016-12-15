#pragma once
#include <string>
#include <Windows.h>

class Protector
{
	static HKEY OpenKey(HKEY hRootKey, LPCSTR strKey);
	void SetVal(LPCTSTR lpValue, DWORD data) const;
	DWORD GetVal(LPCTSTR lpValue) const;
	unsigned launch_count;
	unsigned hash_val;
	static unsigned hash(unsigned x);
	HKEY hKey;
	static Protector* instance;
	Protector();
public:
	static Protector* get_instance();
	bool validate() const;
	~Protector();
};

