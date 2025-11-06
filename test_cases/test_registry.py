import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 80)
    print("Testing Registry")
    print("=" * 80)
    
    from pointcept.engines.simple_train import TRAINERS
    
    print(f"\n📋 Available trainers in registry:")
    print(f"   {list(TRAINERS.module_dict.keys())}")
    
    if "RegressionTrainer" in TRAINERS.module_dict:
        print(f"\n✅ RegressionTrainer is registered!")
        print(f"   Class: {TRAINERS.module_dict['RegressionTrainer']}")
    else:
        print(f"\n❌ RegressionTrainer is NOT registered!")
        print(f"\n🔍 Trying to import manually...")
        
        try:
            from pointcept.engines.simple_train_gelsight import RegressionTrainer
            print(f"   ✅ Import successful: {RegressionTrainer}")
            print(f"   Registered trainers: {list(TRAINERS.module_dict.keys())}")
        except Exception as e:
            print(f"   ❌ Import failed: {e}")
    
    print(f"\n🔧 Trying to build RegressionTrainer...")
    try:
        trainer_cfg = dict(type="RegressionTrainer")
        print(f"   Config: {trainer_cfg}")
        
        if "RegressionTrainer" in TRAINERS.module_dict:
            trainer_cls = TRAINERS.get("RegressionTrainer")
            print(f"   ✅ Successfully got trainer class: {trainer_cls}")
        else:
            print(f"   ❌ Trainer not found in registry")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()